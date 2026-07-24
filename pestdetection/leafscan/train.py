"""Training pipeline: seeded fine-tune -> calibrate -> evaluate -> self-describing bundle.

Reads a YAML config, trains a torchvision backbone with class-weighted CrossEntropy for
the imbalance, selects the best epoch by val macro-F1, fits temperature on the val split,
evaluates on the held-out test split, and writes a self-describing checkpoint + registry
entry + reports. Every artifact needed to verify the run (split CSV, data hash, git sha,
metrics) is captured. Runs are seeded and best-effort deterministic on CPU/CUDA; on MPS
(Apple Silicon) kernels are not bit-reproducible, so metrics reproduce within a tolerance.
"""

from __future__ import annotations

import copy
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .calibration import fit_temperature
from .checkpoint import ModelBundle
from .config import REPO_ROOT
from .data import (
    PestDataset,
    build_split,
    class_weights,
    data_sha256,
    load_split,
    seed_worker,
)
from .evaluate import collect_logits, compute_metrics, save_reports
from .models import build_model
from .preprocess import eval_transform, train_transform
from .registry import update_registry


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    import os

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Best-effort determinism on CUDA/CPU. MPS is not bit-reproducible (see module docstring).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _subset(records: list[tuple[str, int]], max_per_class: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    by_cls: dict[int, list] = {}
    for r in records:
        by_cls.setdefault(r[1], []).append(r)
    out: list[tuple[str, int]] = []
    for _cls, items in by_cls.items():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    return out


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    is_mps = device.type == "mps"
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
        # MPS caches freed blocks and grows over a long run (deeper nets like resnet50 can
        # get OOM-killed mid-epoch). Periodically release the cache to keep memory bounded.
        if is_mps and (i + 1) % 8 == 0:
            torch.mps.empty_cache()
    if is_mps:
        torch.mps.empty_cache()
    return total / max(n, 1)


def _macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(labels, logits.argmax(axis=1), average="macro", zero_division=0))


def train_from_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg.update(overrides)

    seed = int(cfg["seed"])
    set_seed(seed)
    device = resolve_device(cfg.get("device"))

    data_dir = REPO_ROOT / cfg["data_dir"]
    split_csv = REPO_ROOT / cfg["split_csv"]
    if not split_csv.exists() or cfg.get("rebuild_split"):
        build_split(
            data_dir, split_csv, seed=seed,
            ratios=tuple(cfg["split_ratios"]),
            min_eval_per_class=int(cfg.get("min_eval_per_class", 5)),
        )
    class_names, splits = load_split(split_csv)
    num_classes = len(class_names)

    max_per_class = cfg.get("max_per_class")  # smoke/CI knob
    if max_per_class:
        for s in splits:
            splits[s] = _subset(splits[s], int(max_per_class), seed)

    mean, std = cfg["normalize"]["mean"], cfg["normalize"]["std"]
    input_size = int(cfg["input_size"])
    aug = cfg.get("augment", {})
    tf_train = train_transform(
        input_size, mean, std,
        randaugment_ops=int(aug.get("randaugment_ops", 2)),
        randaugment_magnitude=int(aug.get("randaugment_magnitude", 7)),
        horizontal_flip=bool(aug.get("horizontal_flip", True)),
    )
    tf_eval = eval_transform(input_size, mean, std)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=int(cfg.get("num_workers", 4)), worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(
        PestDataset(data_dir, splits["train"], tf_train),
        batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False, **common,
    )
    val_loader = DataLoader(
        PestDataset(data_dir, splits["val"], tf_eval), batch_size=int(cfg["batch_size"]), **common
    )
    test_loader = DataLoader(
        PestDataset(data_dir, splits["test"], tf_eval), batch_size=int(cfg["batch_size"]), **common
    )

    model, gradcam_layer = build_model(cfg["arch"], num_classes, pretrained=bool(cfg.get("pretrained", True)))
    if cfg.get("freeze_backbone"):
        # Linear probe: only the classification head trains.
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad_(False)
    elif cfg.get("freeze_early_stages"):
        # Partial fine-tune: freeze the generic early stages, train the semantic later
        # stages + head. Memory/compute-friendly (fits an 8 GB machine) while still a real
        # fine-tune of a high-capacity backbone.
        frozen = ("conv1", "bn1", "layer1", "layer2")
        for name, p in model.named_parameters():
            if name.startswith(frozen):
                p.requires_grad_(False)
    model.to(device)

    weight = None
    if cfg.get("class_weighted_loss", True):
        weight = class_weights(splits["train"], num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weight, label_smoothing=float(cfg.get("label_smoothing", 0.0))
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg["epochs"])
    scheduler = None
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if scheduler:
            scheduler.step()
        val_logits, val_labels = collect_logits(model, val_loader, device)
        f1 = _macro_f1(val_logits, val_labels)
        print(f"[epoch {epoch}/{epochs}] train_loss={loss:.4f} val_macro_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate temperature on val ONLY (never test), then evaluate test.
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(torch.tensor(val_logits), torch.tensor(val_labels))
    test_logits, test_labels = collect_logits(model, test_loader, device)
    metrics = compute_metrics(test_logits, test_labels, class_names, temperature)
    metrics["name"] = cfg["name"]
    metrics["best_val_macro_f1"] = round(best_f1, 4)
    metrics["device"] = device.type
    metrics["epochs"] = epochs
    metrics["seed"] = seed

    output_dir = REPO_ROOT / cfg["output_dir"]
    reports_dir = REPO_ROOT / cfg.get("reports_dir", "reports")
    save_reports(reports_dir, metrics, test_logits, test_labels, class_names, temperature)

    bundle = ModelBundle(
        arch=cfg["arch"],
        class_names=class_names,
        mean=mean,
        std=std,
        input_size=input_size,
        gradcam_layer=gradcam_layer,
        temperature=temperature,
        abstain_threshold=float(cfg.get("abstain_threshold", 0.40)),
        data_sha256=data_sha256(data_dir),
        git_sha=git_sha(),
        metrics=metrics,
    )
    bundle.save(output_dir, model.cpu())

    update_registry(
        REPO_ROOT / "models" / "registry.json",
        {
            "name": cfg["name"],
            "arch": cfg["arch"],
            "path": str(Path(cfg["output_dir"])),
            "weights_sha256": bundle.weights_sha256,
            "class_count": num_classes,
            "created_at": bundle.created_at,
            "git_sha": bundle.git_sha,
            "metrics": {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "top3_accuracy": metrics["top3_accuracy"],
                "ece_after": metrics["ece_after"],
            },
        },
    )

    print(
        f"DONE: macro_f1={metrics['macro_f1']} accuracy={metrics['accuracy']} "
        f"top3={metrics['top3_accuracy']} T={temperature:.3f} "
        f"ECE {metrics['ece_before']}->{metrics['ece_after']}",
        flush=True,
    )
    return metrics
