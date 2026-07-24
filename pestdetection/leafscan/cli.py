"""``leafscan`` command-line interface.

Subcommands:
  split     build the deterministic stratified split CSV
  train     train -> calibrate -> evaluate -> write self-describing bundle + reports
  evaluate  evaluate an existing bundle on the test split
  predict   run inference on a single image
  card      (re)generate MODEL_CARD.md from a bundle
"""

from __future__ import annotations

import argparse
import json

from .config import REPO_ROOT


def _cmd_split(args) -> int:
    from .data import build_split
    from .train import load_config

    cfg = load_config(args.config)
    counts = build_split(
        REPO_ROOT / cfg["data_dir"],
        REPO_ROOT / cfg["split_csv"],
        seed=int(cfg["seed"]),
        ratios=tuple(cfg["split_ratios"]),
        min_eval_per_class=int(cfg.get("min_eval_per_class", 5)),
    )
    print("split counts:", counts)
    return 0


def _cmd_train(args) -> int:
    from .train import train_from_config

    overrides = {}
    if args.smoke:
        overrides = {
            "epochs": 1,
            "max_per_class": 12,
            "freeze_backbone": True,
            "num_workers": 0,
            "pretrained": args.pretrained,
            "output_dir": "models/_smoke",
            "reports_dir": "reports/_smoke",
            "name": "pest_smoke",
            "device": args.device,
        }
    elif args.device:
        overrides["device"] = args.device
    train_from_config(args.config, overrides=overrides)
    return 0


def _cmd_evaluate(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from .calibration import fit_temperature
    from .checkpoint import ModelBundle
    from .data import PestDataset, load_split, seed_worker
    from .evaluate import collect_logits, compute_metrics, save_reports
    from .preprocess import eval_transform
    from .train import load_config, resolve_device

    cfg = load_config(args.config)
    bundle = ModelBundle.load(args.model_dir)
    device = resolve_device(args.device)
    bundle.model.to(device)
    class_names, splits = load_split(REPO_ROOT / cfg["split_csv"])
    tf = eval_transform(bundle.input_size, bundle.mean, bundle.std)
    data_dir = REPO_ROOT / cfg["data_dir"]
    loaders = {
        s: DataLoader(PestDataset(data_dir, splits[s], tf), batch_size=32, worker_init_fn=seed_worker)
        for s in ("val", "test")
    }
    vlogits, vlabels = collect_logits(bundle.model, loaders["val"], device)
    temperature = fit_temperature(torch.tensor(vlogits), torch.tensor(vlabels))
    tlogits, tlabels = collect_logits(bundle.model, loaders["test"], device)
    metrics = compute_metrics(tlogits, tlabels, class_names, temperature)
    save_reports(REPO_ROOT / cfg.get("reports_dir", "reports"), metrics, tlogits, tlabels, class_names, temperature)
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "accuracy", "top3_accuracy", "ece_before", "ece_after")}, indent=2))
    return 0


def _cmd_predict(args) -> int:
    from PIL import Image

    from .inference import Predictor

    predictor = Predictor(model_dir=args.model_dir)
    pred = predictor.predict(Image.open(args.image), want_gradcam=False)
    print(json.dumps({
        "prediction": pred.display_label,
        "confidence": pred.confidence,
        "severity": pred.severity,
        "abstained": pred.abstained,
        "top_predictions": pred.top_predictions,
    }, indent=2))
    return 0


def _cmd_card(args) -> int:
    from .modelcard import write_model_card

    out = args.out or (REPO_ROOT / "MODEL_CARD.md")
    write_model_card(args.model_dir, out)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="leafscan", description="LeafScan pest classifier toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    default_cfg = str(REPO_ROOT / "configs" / "pest_resnet50.yaml")
    default_model = str(REPO_ROOT / "models" / "pest_resnet50")

    p = sub.add_parser("split", help="build the stratified split CSV")
    p.add_argument("--config", default=default_cfg)
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("train", help="train -> calibrate -> evaluate -> bundle")
    p.add_argument("--config", default=default_cfg)
    p.add_argument("--smoke", action="store_true", help="tiny 1-epoch run for CI/verification")
    p.add_argument("--pretrained", action="store_true", help="(smoke) use pretrained backbone")
    p.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", help="evaluate an existing bundle on the test split")
    p.add_argument("--config", default=default_cfg)
    p.add_argument("--model-dir", default=default_model)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("predict", help="predict a single image")
    p.add_argument("image")
    p.add_argument("--model-dir", default=default_model)
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("card", help="regenerate MODEL_CARD.md")
    p.add_argument("--model-dir", default=default_model)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_card)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
