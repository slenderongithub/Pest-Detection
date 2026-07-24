#!/usr/bin/env python
"""Hash-verified checkpoint fetcher.

Replaces the old broken ``download_model.py`` (which imported a nonexistent symbol).
Reads ``models/registry.json`` for the requested model, downloads its ``weights.pt``
from the URL recorded in the registry (or ``--url`` / ``$LEAFSCAN_WEIGHTS_URL``), and
verifies the SHA256 against the committed manifest before writing it into place.

The committed manifest.json (taxonomy + metrics + weights_sha256) is what makes this
safe and verifiable even though weights.pt is gitignored.

Usage:
    python scripts/fetch_model.py                      # fetch the default model
    python scripts/fetch_model.py --name pest_resnet18 --url https://.../weights.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify model weights.")
    parser.add_argument("--name", default="pest_resnet50")
    parser.add_argument("--url", default=None, help="weights.pt URL (else registry/env)")
    parser.add_argument("--registry", default=str(REPO_ROOT / "models" / "registry.json"))
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"error: no registry at {registry_path}", file=sys.stderr)
        return 2
    registry = json.loads(registry_path.read_text())
    entry = next((m for m in registry.get("models", []) if m.get("name") == args.name), None)
    if entry is None:
        print(f"error: model {args.name!r} not in registry", file=sys.stderr)
        return 2

    model_dir = REPO_ROOT / entry["path"]
    manifest = json.loads((model_dir / "manifest.json").read_text())
    expected_sha = manifest.get("weights_sha256") or entry.get("weights_sha256")
    weights_path = model_dir / "weights.pt"

    if weights_path.exists() and expected_sha and sha256_file(weights_path) == expected_sha:
        print(f"✓ {weights_path} already present and verified.")
        return 0

    import os

    url = args.url or entry.get("weights_url") or os.environ.get("LEAFSCAN_WEIGHTS_URL")
    if not url:
        print(
            "error: no weights URL. Provide --url, set $LEAFSCAN_WEIGHTS_URL, or add\n"
            "       a 'weights_url' to the registry entry. (Train locally with\n"
            "       `leafscan train` to produce weights without downloading.)",
            file=sys.stderr,
        )
        return 2

    print(f"Downloading {args.name} weights from {url} …")
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = weights_path.with_suffix(".pt.part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256_file(tmp)
    if expected_sha and got != expected_sha:
        tmp.unlink(missing_ok=True)
        print(f"error: SHA256 mismatch\n  expected {expected_sha}\n  got      {got}", file=sys.stderr)
        return 1
    tmp.replace(weights_path)
    print(f"✓ wrote {weights_path} (sha256 {got[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
