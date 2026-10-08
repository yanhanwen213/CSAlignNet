"""
Fixed-pose sweep evaluation (rot_only / shift_only / combo) with Acc / F1 / EER.

Saves per model / fold:
  results/<model>/ptfold<k>/<mode>/{metrics.csv,metrics.json,summary.json}
  log/eval_<stamp>/config.json + metrics.csv/jsonl

Example:
  python pose_sweep.py --model_type csalignnet ^
    --ckpt <checkpoints>/fold0/cs_align_best.pth --fold 0 ^
    --data_root <data-root> --modes rot_only shift_only combo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
_CSALIGNNET = REPO / "csalignnet"
_LRFV = Path(__import__("os").environ.get("LRFVNET_PATH", str(REPO.parent / "LRFVNet")))
# Shared helpers live in csalignnet (kfold/logger live in evaluation/).
for p in (_CSALIGNNET, _LRFV, ROOT):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lrfvnet.dataset import FingerVeinFolder, build_default_transform  # noqa: E402
from lrfvnet.metrics import (  # noqa: E402
    evaluate_classification_and_biometrics,
    extract_embeddings,
)

from kfold import split_kfold  # noqa: E402  (evaluation)
from logger import EpochLogger  # noqa: E402  (csalignnet)
from load_model import load_model  # noqa: E402


def _pil_rotate_translate(img: Image.Image, angle: float, dx: float, dy: float) -> Image.Image:
    out = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return out.transform(
        out.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=0,
    )


class FixedPoseDataset(Dataset):
    """Val images with a fixed rotation/translation (PIL, same as training aug)."""

    def __init__(
        self,
        root: str | Path,
        indices: list[int],
        angle: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        image_size: int = 112,
        transform=None,
    ) -> None:
        self.base = FingerVeinFolder(root, transform=None)
        self.indices = list(indices)
        self.angle = float(angle)
        self.dx = float(dx)
        self.dy = float(dy)
        self.image_size = image_size
        self.transform = transform or build_default_transform(image_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_i = self.indices[index]
        path, label = self.base.samples[real_i]
        # Level-0 fix: geometry BEFORE resize (same as cs_dataset training aug).
        # Old code resized to 112 first then rotated -> black corners + blur
        # ate thin veins at rot>=20deg. Native-rotate-then-resize restores them.
        img = Image.open(path).convert("L")
        if self.angle != 0.0 or self.dx != 0.0 or self.dy != 0.0:
            img = _pil_rotate_translate(img, self.angle, self.dx, self.dy)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.transform(img), int(label)


def build_settings(mode: str, angles: list[float], shifts: list[float]) -> list[dict]:
    settings = []
    if mode == "rot_only":
        for a in angles:
            settings.append({"setting": f"rot_{a}", "angle": a, "dx": 0.0, "dy": 0.0})
    elif mode == "shift_only":
        for s in shifts:
            settings.append({"setting": f"shift_{s}", "angle": 0.0, "dx": s, "dy": 0.0})
    elif mode == "combo":
        n = max(len(angles), len(shifts))
        for i in range(n):
            a = angles[min(i, len(angles) - 1)]
            s = shifts[min(i, len(shifts) - 1)]
            settings.append(
                {"setting": f"a{a}_x{s}_y0.0", "angle": a, "dx": s, "dy": 0.0}
            )
    else:
        raise ValueError(mode)
    return settings


def save_mode_results(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Keep legacy "mean" (=acc) for old plot scripts; add f1/eer.
    summary = {}
    for r in rows:
        key = r["setting"]
        summary[key] = {
            "mean": r["val_acc"],
            "acc_mean": r["val_acc"],
            "f1_mean": r["val_f1"],
            "eer_mean": r["val_eer"],
            "std": 0.0,
            "n": 1,
        }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Pose sweep eval with Acc/F1/EER")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["pida", "lrfv", "lrfvnet", "b3", "b4", "b5"],
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--modes", type=str, nargs="+", default=["rot_only", "shift_only", "combo"])
    p.add_argument("--angles", type=float, nargs="+", default=[0, 10, 20, 30, 35])
    p.add_argument("--shifts", type=float, nargs="+", default=[0, 10, 20, 30, 40])
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else (ROOT / "results")
    log_root = Path(args.log_dir) if args.log_dir else (ROOT / "log")

    raw_type = args.model_type.lower()
    if raw_type == "pida":
        model_type = "pida"
    elif raw_type == "b3":
        model_type = "b3"
    elif raw_type == "b4":
        model_type = "b4"
    elif raw_type == "b5":
        model_type = "b5"
    else:
        model_type = "lrfv"
    model, meta = load_model(args.ckpt, model_type, device)
    if model_type in ("pida", "b3", "b4"):
        num_classes = int(model.recognizer.num_classes)
    else:
        num_classes = int(model.num_classes)

    full = FingerVeinFolder(data_root, transform=None)
    train_idx, val_idx = split_kfold(full, fold=args.fold, k=args.kfold, seed=args.seed)

    def get_logits_emb(m, images):
        if model_type in ("pida", "b3", "b4"):
            aux = m(images, return_aux=True)
            # Angular-trained checkpoints: the LoRaLin head never received
            # gradients (random weights), so acc must use the trained angular
            # head in eval mode (labels=None -> s*cos, no margin added).
            if getattr(m, "use_angular", False) and getattr(m, "angular_head", None) is not None:
                try:
                    return m.angular_logits(aux["embedding"], None), aux["embedding"]
                except Exception:
                    pass
            return aux["logits"], aux["embedding"]
        return m(images, return_embedding=True)

    gal_loader = DataLoader(
        FixedPoseDataset(data_root, train_idx, angle=0.0, dx=0.0, dy=0.0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    gal_emb, gal_y = extract_embeddings(model, gal_loader, device, get_logits_emb)
    print(f"Gallery embeddings: {gal_emb.shape[0]} (train split, clean; eer=probe–gallery)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logger = EpochLogger(
        log_root, run_name=f"eval_{model_type}_fold{args.fold}_{stamp}"
    )
    run_logger.save_config(
        {
            **vars(args),
            "eer_protocol": "probe_gallery",
            "ckpt_meta": {k: meta[k] for k in ("epoch", "acc", "f1", "eer", "fold")},
            "num_train_gallery": len(train_idx),
            "num_val": len(val_idx),
            "num_classes": num_classes,
        }
    )
    print(f"Eval log -> {run_logger.path()}")

    tag = {"pida": "pida", "b3": "B3", "b4": "B4", "b5": "B5", "lrfv": "LRFV"}[model_type]
    fold_out = out_root / tag / f"ptfold{args.fold}"

    epoch_i = 0
    for mode in args.modes:
        settings = build_settings(mode, args.angles, args.shifts)
        rows = []
        for s in settings:
            ds = FixedPoseDataset(
                data_root,
                val_idx,
                angle=s["angle"],
                dx=s["dx"],
                dy=s["dy"],
            )
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            metrics = evaluate_classification_and_biometrics(
                model,
                loader,
                device,
                num_classes=num_classes,
                get_logits_and_emb=get_logits_emb,
                gallery_embeddings=gal_emb,
                gallery_labels=gal_y,
            )
            # Level-0 fix: Rank-1 via cosine templates (same as cs_metrics),
            # immune to classifier-index mismatch. Probe=posed val,
            # Gallery=clean train split. EER from metrics already uses
            # probe-gallery protocol; Rank-1 added here for the paper curve.
            from lrfvnet.metrics import _l2_normalize as _l2n

            import numpy as np

            @torch.no_grad()
            def _probe_embs():
                model.eval()
                es, ys = [], []
                for images, labels in loader:
                    images = images.to(device)
                    _, e = get_logits_emb(model, images)
                    es.append(e.detach().cpu().numpy())
                    ys.append(labels.numpy())
                return np.concatenate(es, 0), np.concatenate(ys, 0)

            p_emb, p_y = _probe_embs()
            g_n = _l2n(np.asarray(gal_emb, dtype=np.float64))
            p_n = _l2n(np.asarray(p_emb, dtype=np.float64))
            classes = sorted(set(gal_y.tolist()))
            tmpl = np.stack([g_n[gal_y == c].mean(0) for c in classes], 0)
            tmpl = tmpl / (np.linalg.norm(tmpl, axis=1, keepdims=True) + 1e-12)
            tmpl_y = np.asarray(classes)
            nn_idx = (p_n @ tmpl.T).argmax(1)
            rank1 = float((tmpl_y[nn_idx] == p_y).mean() * 100.0)
            row = {
                "model_type": model_type,
                "fold": args.fold,
                "ckpt": str(Path(args.ckpt).resolve()),
                "setting": s["setting"],
                "angle": s["angle"],
                "dx": s["dx"],
                "dy": s["dy"],
                "val_acc": metrics["acc"],
                "val_f1": metrics["f1"],
                "val_eer": metrics["eer"],
                "rank1": round(rank1, 4),
                "ckpt_epoch": meta.get("epoch"),
                "ckpt_acc": meta.get("acc"),
            }
            rows.append(row)
            epoch_i += 1
            msg = (
                f"[{mode}] {s['setting']}: acc={metrics['acc']:.2f}% "
                f"f1={metrics['f1']:.2f}% eer={metrics['eer']:.4f}% "
                f"rank1={rank1:.2f}%"
            )
            print(msg)
            run_logger.log_epoch(epoch_i, {"mode": mode, **row}, message=msg)

        save_mode_results(fold_out / mode, rows)

    print(f"Results saved under -> {fold_out}")


if __name__ == "__main__":
    main()
