"""Tiny JSON model registry.

Answers the question this project could not previously answer: "which model is deployed,
and how good is it?" One committed ``models/registry.json`` maps a model name to its
path, weights hash, class count, headline metrics, and creation date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    p = Path(registry_path)
    if not p.exists():
        return {"models": []}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def update_registry(registry_path: str | Path, entry: dict[str, Any]) -> None:
    """Upsert ``entry`` (keyed by ``name``) into the registry file."""
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    reg = load_registry(p)
    models = [m for m in reg.get("models", []) if m.get("name") != entry.get("name")]
    models.append(entry)
    reg["models"] = sorted(models, key=lambda m: m.get("name", ""))
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)
