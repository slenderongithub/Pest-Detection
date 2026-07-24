"""Pest knowledge base loader.

Replaces the old disease-keyed ``DISEASE_INFO`` dict (which would have missed every
prediction from the pest model). Knowledge is data, loaded from
``data/knowledge/pests.yaml`` and keyed on the exact 9 pest class names, with an honest
``None`` fallback for classes that have no entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import REPO_ROOT

KNOWLEDGE_PATH = REPO_ROOT / "data" / "knowledge" / "pests.yaml"

SEVERITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


@dataclass(frozen=True)
class Pesticide:
    name: str
    dose: str


@dataclass(frozen=True)
class PestInfo:
    class_name: str
    common_name: str
    pest_type: str
    severity: str
    description: str
    pesticides: list[Pesticide] = field(default_factory=list)
    ipm: str = ""

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, -1)


@lru_cache(maxsize=1)
def _raw(path: str | None = None) -> dict:
    p = Path(path) if path else KNOWLEDGE_PATH
    if not p.exists():
        return {"_meta": {}, "pests": {}}
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def disclaimer() -> str:
    return _raw().get("_meta", {}).get("disclaimer", "").strip()


def get_pest_info(class_name: str) -> PestInfo | None:
    """Return structured knowledge for a class name, or ``None`` if there is no entry.

    Matching is exact on the class name first (the industry-standard, unambiguous path),
    then falls back to a case-insensitive match so minor label casing differences still
    resolve rather than silently returning ``None``.
    """
    pests = _raw().get("pests", {}) or {}
    entry = pests.get(class_name)
    if entry is None:
        lowered = {k.lower(): k for k in pests}
        key = lowered.get(class_name.lower())
        entry = pests.get(key) if key else None
    if entry is None:
        return None
    return PestInfo(
        class_name=class_name,
        common_name=str(entry.get("common_name", class_name)),
        pest_type=str(entry.get("pest_type", "Unknown")),
        severity=str(entry.get("severity", "Medium")),
        description=str(entry.get("description", "")).strip(),
        pesticides=[
            Pesticide(name=str(p.get("name", "")), dose=str(p.get("dose", "")))
            for p in entry.get("pesticides", []) or []
        ],
        ipm=str(entry.get("ipm", "")).strip(),
    )


def pretty_label(label: str) -> str:
    """Human-friendly display form of a class name (e.g. capitalise, keep spacing)."""
    return label.replace("_", " ").strip()
