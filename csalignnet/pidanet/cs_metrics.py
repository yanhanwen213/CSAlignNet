"""Cross-session matching metrics (gallery=S1, probe=S2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_LRFV = Path(__file__).resolve().parents[2] / "LRFVNet"
if _LRFV.is_dir() and str(_LRFV) not in sys.path:
    sys.path.insert(0, str(_LRFV))

from lrfvnet.metrics import _eer_from_score_arrays, _macro_f1  # noqa: E402


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    logits_all, emb_all, sess_all, y_all = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        aux = model(images, return_aux=True)
        # Angular models: LoRaLin head never gets gradients (random weights).
        # Eval acc must use the trained angular head (s*cos, no margin).
        if getattr(model, "use_angular", False) and getattr(model, "angular_head", None) is not None:
            try:
                logits_all.append(model.angular_logits(aux["embedding"], None).cpu())
            except Exception:
                logits_all.append(aux["logits"].cpu())
        else:
            logits_all.append(aux["logits"].cpu())
        emb_all.append(aux["embedding"].cpu())
        sess_all.append(aux["session_embedding"].cpu())
        y_all.append(labels.cpu())
    return (
        torch.cat(logits_all, 0).numpy(),
        torch.cat(emb_all, 0).numpy(),
        torch.cat(sess_all, 0).numpy(),
        torch.cat(y_all, 0).numpy(),
    )


def _templates(emb: np.ndarray, y: np.ndarray):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    classes = sorted(set(y.tolist()))
    templates = np.stack([e[y == c].mean(0) for c in classes], 0)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12
    return templates, np.asarray(classes)


def rank1_identification(gallery_emb, gallery_y, probe_emb, probe_y) -> float:
    templates, tmpl_y = _templates(gallery_emb, gallery_y)
    p = probe_emb / (np.linalg.norm(probe_emb, axis=1, keepdims=True) + 1e-12)
    nn = (p @ templates.T).argmax(1)
    return float((tmpl_y[nn] == probe_y).mean() * 100.0)


def probe_gallery_eer(
    gallery_emb, gallery_y, probe_emb, probe_y, max_impostor: int = 200000
) -> float:
    templates, tmpl_y = _templates(gallery_emb, gallery_y)
    p = probe_emb / (np.linalg.norm(probe_emb, axis=1, keepdims=True) + 1e-12)
    genuine, impostor = [], []
    rng = np.random.RandomState(0)
    for i, yi in enumerate(probe_y):
        scores = p[i] @ templates.T
        hit = np.where(tmpl_y == yi)[0]
        if len(hit):
            genuine.append(float(scores[hit[0]]))
        miss = np.where(tmpl_y != yi)[0]
        impostor.extend(scores[miss].tolist())
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if len(impostor) > max_impostor:
        impostor = impostor[rng.choice(len(impostor), max_impostor, replace=False)]
    return float(_eer_from_score_arrays(genuine, impostor))


@torch.no_grad()
def evaluate_cross_session(
    model,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    device,
    num_classes: int,
) -> dict:
    """Closed-set probe Acc/F1 + Rank-1/EER for backbone & session embeddings."""
    _, g_emb, g_sess, g_y = extract_features(model, gallery_loader, device)
    p_logits, p_emb, p_sess, p_y = extract_features(model, probe_loader, device)

    pred = p_logits.argmax(1)
    acc = float((pred == p_y).mean() * 100.0)
    f1 = float(_macro_f1(p_y, pred, num_classes))

    r1_emb = rank1_identification(g_emb, g_y, p_emb, p_y)
    eer_emb = probe_gallery_eer(g_emb, g_y, p_emb, p_y)
    r1_sess = rank1_identification(g_sess, g_y, p_sess, p_y)
    eer_sess = probe_gallery_eer(g_sess, g_y, p_sess, p_y)

    return {
        "probe_cls_acc": acc,
        "probe_cls_f1": f1,
        "rank1_embedding": r1_emb,
        "eer_embedding": eer_emb,
        "rank1_session": r1_sess,
        "eer_session": eer_sess,
    }
