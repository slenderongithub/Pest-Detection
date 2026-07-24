"""The knowledge base must cover every class the model can predict (the trap the old
disease-keyed dict fell into), and its keys must match the dataset folder names."""

from __future__ import annotations

from leafscan import knowledge as K
from leafscan.config import REPO_ROOT


def test_every_class_has_knowledge(class_names):
    for cls in class_names:
        info = K.get_pest_info(cls)
        assert info is not None, f"no knowledge entry for {cls!r}"
        assert info.severity in {"Low", "Medium", "High", "Critical"}
        assert info.pesticides, f"{cls} has no pesticide references"


def test_missing_class_returns_none():
    assert K.get_pest_info("definitely not a pest") is None


def test_case_insensitive_fallback():
    assert K.get_pest_info("APHIDS") is not None


def test_disclaimer_present():
    assert "prescription" in K.disclaimer().lower()


def test_knowledge_keys_match_dataset_folders():
    data_dir = REPO_ROOT / "data" / "Pest_Dataset"
    if not data_dir.exists():
        return  # dataset is gitignored; skip when absent
    folders = {p.name for p in data_dir.iterdir() if p.is_dir()}
    known = set(K._raw().get("pests", {}))
    assert folders <= known, f"classes missing from knowledge base: {folders - known}"
