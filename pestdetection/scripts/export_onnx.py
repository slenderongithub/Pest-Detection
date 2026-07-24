"""Export a LeafScan checkpoint to ONNX for fully in-browser (onnxruntime-web) inference.

Produces everything the static Netlify site needs, into ``web/model/``:
  - model.onnx        : two outputs — logits (1x9) AND feat (1x2048x7x7, the layer4
                        activation) so the browser can compute a gradient-free CAM.
  - fc.bin            : float32 fc weight matrix (num_classes x 2048), used for CAM.
  - meta.json         : class names, normalization, input size, temperature, abstain
                        threshold, fc shape, knowledge base, disclaimer — the whole
                        serving contract, read from the checkpoint manifest (never
                        hardcoded), so the browser matches the Python pipeline exactly.

There is NO autograd in the browser, so Grad-CAM is replaced by classic CAM. For a ResNet
(global-avg-pool → fc) the two are equivalent up to a positive scale:
    CAM_c[h,w] = ReLU( sum_k fc.weight[c,k] * feat[k,h,w] ).

Run:  python scripts/export_onnx.py [--model-dir models/pest_resnet50] [--quantize]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from leafscan.checkpoint import ModelBundle
from leafscan.config import REPO_ROOT
from leafscan.knowledge import KNOWLEDGE_PATH, disclaimer


class LogitsAndFeat(nn.Module):
    """Wrap a torchvision ResNet to also return the pre-pool layer4 activation (for CAM)."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):  # noqa: D401 - explicit resnet forward so ONNX traces both outputs
        m = self.m
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        feat = x
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        logits = m.fc(x)
        return logits, feat


def export(model_dir: str, out_dir: Path, quantize: bool) -> None:
    bundle = ModelBundle.load(model_dir)
    model = bundle.model.eval()
    if bundle.arch not in ("resnet18", "resnet34", "resnet50"):
        raise SystemExit(f"CAM export assumes a torchvision ResNet; got {bundle.arch!r}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    wrapped = LogitsAndFeat(model).eval()
    dummy = torch.zeros(1, 3, bundle.input_size, bundle.input_size)
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits", "feat"],
        opset_version=17,
        dynamic_axes=None,  # fixed 1x3xSxS — one image at a time in the browser
    )

    # --- Parity check: torch logits must match onnxruntime logits on a real sample. ---
    import onnxruntime as ort

    sample = _sample_tensor(bundle)
    with torch.no_grad():
        t_logits, t_feat = wrapped(sample)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_logits, o_feat = sess.run(None, {"input": sample.numpy()})
    logit_diff = float(np.abs(t_logits.numpy() - o_logits).max())
    feat_diff = float(np.abs(t_feat.numpy() - o_feat).max())
    print(f"[parity] max|logits torch-onnx| = {logit_diff:.2e}   max|feat| = {feat_diff:.2e}")
    assert logit_diff < 1e-3, f"ONNX logits diverge from torch ({logit_diff}); export is wrong."

    # --- fc weights for CAM (float32, shape num_classes x 2048). ---
    fc_w = model.fc.weight.detach().cpu().numpy().astype("<f4")
    (out_dir / "fc.bin").write_bytes(fc_w.tobytes())

    # --- Optional static int8 quantization (smaller browser download). ---
    served = "model.onnx"
    if quantize:
        served = _quantize(onnx_path, out_dir, bundle, sess)
        if served != "model.onnx":
            onnx_path.unlink(missing_ok=True)  # keep the deploy dir lean: only ship the served model

    # --- Serving contract + knowledge, straight from the checkpoint/knowledge base. ---
    import yaml

    knowledge = yaml.safe_load(KNOWLEDGE_PATH.read_text(encoding="utf-8")) or {}
    meta = {
        "model_name": bundle.metrics.get("name") if bundle.metrics else bundle.arch,
        "arch": bundle.arch,
        "onnx": served,
        "class_names": list(bundle.class_names),
        "mean": list(bundle.mean),
        "std": list(bundle.std),
        "input_size": bundle.input_size,
        "resize_ratio": 256 / 224,
        "temperature": float(bundle.temperature),
        "abstain_threshold": float(bundle.abstain_threshold),
        "fc_shape": list(fc_w.shape),  # [num_classes, channels]
        "macro_f1": (bundle.metrics or {}).get("macro_f1"),
        "pests": knowledge.get("pests", {}),
        "disclaimer": disclaimer(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = (out_dir / served).stat().st_size / 1e6
    print(f"[done] wrote {out_dir}/  (served model: {served}, {size:.0f} MB)")
    print(f"       classes={len(bundle.class_names)}  input={bundle.input_size}  T={bundle.temperature:.4f}")


def _sample_tensor(bundle: ModelBundle) -> torch.Tensor:
    """A real preprocessed dataset image if available, else deterministic noise."""
    from PIL import Image

    from leafscan.preprocess import preprocess_image

    data_dir = REPO_ROOT / "data" / "Pest_Dataset"
    imgs = sorted(data_dir.rglob("*.jpg"))[:1] if data_dir.exists() else []
    if imgs:
        return preprocess_image(Image.open(imgs[0]), bundle.input_size, bundle.mean, bundle.std)
    torch.manual_seed(0)
    return torch.randn(1, 3, bundle.input_size, bundle.input_size)


def _quantize(onnx_path: Path, out_dir: Path, bundle: ModelBundle, fp32_sess) -> str:
    """Static int8 quantization with calibration from the val split; verify accuracy holds."""
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    except Exception as e:  # pragma: no cover
        print(f"[quantize] skipped — onnxruntime.quantization unavailable ({e}). Serving fp32.")
        return "model.onnx"

    from PIL import Image

    from leafscan.preprocess import preprocess_image

    data_dir = REPO_ROOT / "data" / "Pest_Dataset"
    cal_imgs = sorted(data_dir.rglob("*.jpg"))[:80] if data_dir.exists() else []
    if len(cal_imgs) < 16:
        print("[quantize] skipped — need dataset images for calibration. Serving fp32.")
        return "model.onnx"

    class Reader(CalibrationDataReader):
        def __init__(self, paths):
            self._it = iter(paths)

        def get_next(self):
            p = next(self._it, None)
            if p is None:
                return None
            t = preprocess_image(Image.open(p), bundle.input_size, bundle.mean, bundle.std)
            return {"input": t.numpy()}

    q_path = out_dir / "model.int8.onnx"
    quantize_static(str(onnx_path), str(q_path), Reader(cal_imgs), weight_type=QuantType.QInt8)

    # Accuracy sanity: quantized top-1 should agree with fp32 on most calibration images.
    import onnxruntime as ort

    qsess = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
    agree = 0
    for p in cal_imgs[:40]:
        x = preprocess_image(Image.open(p), bundle.input_size, bundle.mean, bundle.std).numpy()
        f = int(np.argmax(fp32_sess.run(None, {"input": x})[0]))
        q = int(np.argmax(qsess.run(None, {"input": x})[0]))
        agree += f == q
    rate = agree / 40
    size = q_path.stat().st_size / 1e6
    print(f"[quantize] int8 top-1 agrees with fp32 on {rate:.0%} of samples ({size:.0f} MB)")
    if rate < 0.9:
        print("[quantize] agreement <90% — keeping fp32 as the served model.")
        q_path.unlink(missing_ok=True)
        return "model.onnx"
    return "model.int8.onnx"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/pest_resnet50")
    ap.add_argument("--out", default=str(REPO_ROOT / "web" / "model"))
    ap.add_argument("--quantize", action="store_true", help="also emit an int8 model (~4x smaller)")
    a = ap.parse_args()
    export(a.model_dir, Path(a.out), a.quantize)
