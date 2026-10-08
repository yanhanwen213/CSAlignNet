"""
Stratified K-fold split — same protocol as LRFVNet (paper / LRFVNet/train.py).

Per class: shuffle samples, then idxs[i::k] -> fold i.
Fold `fold` is validation; the other k-1 folds are training.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path


def split_kfold(dataset, fold: int, k: int = 6, seed: int = 42):
    """
    Args:
        dataset: FingerVeinFolder-like with `.class_to_indices`
        fold: validation fold in [0, k)
    Returns:
        train_indices, val_indices
    """
    if not (0 <= fold < k):
        raise ValueError(f"fold must be in [0, {k}), got {fold}")

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for _, idxs in sorted(dataset.class_to_indices.items()):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        folds = [idxs[i::k] for i in range(k)]
        for i, part in enumerate(folds):
            if i == fold:
                val_idx.extend(part)
            else:
                train_idx.extend(part)
    return train_idx, val_idx


def build_all_folds(dataset, k: int = 6, seed: int = 42) -> list[dict]:
    """Return list of {fold, train_indices, val_indices} for fold=0..k-1."""
    out = []
    for fold in range(k):
        tr, va = split_kfold(dataset, fold=fold, k=k, seed=seed)
        out.append(
            {
                "fold": fold,
                "train_indices": [int(i) for i in tr],
                "val_indices": [int(i) for i in va],
                "num_train": len(tr),
                "num_val": len(va),
            }
        )
    return out


def save_kfold_split(
    dataset,
    out_path: Path,
    k: int = 6,
    seed: int = 42,
) -> dict:
    folds = build_all_folds(dataset, k=k, seed=seed)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "LRFVNet-style stratified K-fold",
        "data_root": str(Path(dataset.root).resolve()),
        "k": k,
        "seed": seed,
        "num_classes": len(dataset.classes),
        "classes": list(dataset.classes),
        "folds": folds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_kfold_split(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
