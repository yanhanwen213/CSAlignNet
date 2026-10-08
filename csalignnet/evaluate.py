"""
Cross-session evaluation: gallery = Session1, probe = Session2.

Reports:
  - closed-set Acc / macro-F1 on probe (logits)
  - Rank-1 / EER for backbone embedding AND session embedding
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# External LRFVNet backbone: <repo-parent>/LRFVNet (or set LRFVNET_PATH).
_LRFV = Path(os.environ.get("LRFVNET_PATH", str(ROOT.parent.parent / "LRFVNet")))
if _LRFV.is_dir() and str(_LRFV) not in sys.path:
    sys.path.insert(0, str(_LRFV))

from pidanet.cs_dataset import DualSessionIndex, SessionImageDataset
from pidanet.cs_metrics import evaluate_cross_session
from pidanet.cs_model import CSAlignNet


def parse_args():
    p = argparse.ArgumentParser(description="S1 gallery → S2 probe evaluation")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--session1_root", type=str, required=True)
    p.add_argument("--session2_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CLAHE; default reads from ckpt args",
    )
    p.add_argument("--clahe_clip_limit", type=float, default=None)
    p.add_argument("--clahe_tile_grid", type=int, nargs=2, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    class_names = ckpt.get("class_names")
    index = DualSessionIndex(args.session1_root, args.session2_root)
    if class_names is None:
        class_names = index.class_names
    num_classes = ckpt.get("num_classes") or len(class_names)

    use_clahe = (
        bool(args.clahe)
        if args.clahe is not None
        else bool(ckpt_args.get("clahe", False))
    )
    clip_limit = (
        float(args.clahe_clip_limit)
        if args.clahe_clip_limit is not None
        else float(ckpt_args.get("clahe_clip_limit", 2.0))
    )
    tile = (
        tuple(args.clahe_tile_grid)
        if args.clahe_tile_grid is not None
        else tuple(ckpt_args.get("clahe_tile_grid", [8, 8]))
    )

    has_dom = any(k.startswith("domain_clf.") for k in ckpt["model"])
    use_domain_adv = bool(ckpt_args.get("domain_adv", False)) and has_dom
    model = CSAlignNet(
        num_classes=num_classes,
        warp_mode=ckpt_args.get("warp_mode", "rt"),
        tps_grid=ckpt_args.get("tps_grid", 3),
        t_scale=ckpt_args.get("t_scale", 0.5),
        session_hidden=ckpt_args.get("session_hidden", 256),
        session_out=ckpt_args.get("session_out", 128),
        use_domain_adv=use_domain_adv or has_dom,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    class_to_idx = {n: i for i, n in enumerate(class_names)}
    gallery_set = SessionImageDataset(
        args.session1_root,
        class_names=class_names,
        class_to_idx=class_to_idx,
        orig_only=True,
        use_clahe=use_clahe,
        clahe_clip_limit=clip_limit,
        clahe_tile_grid=tile,
    )
    probe_set = SessionImageDataset(
        args.session2_root,
        class_names=class_names,
        class_to_idx=class_to_idx,
        orig_only=True,
        use_clahe=use_clahe,
        clahe_clip_limit=clip_limit,
        clahe_tile_grid=tile,
    )
    gallery_loader = DataLoader(
        gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    probe_loader = DataLoader(
        probe_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    m = evaluate_cross_session(
        model, gallery_loader, probe_loader, device, num_classes=num_classes
    )
    result = {
        "protocol": "gallery=session1, probe=session2",
        "ckpt": str(args.ckpt),
        "clahe": use_clahe,
        "clahe_clip_limit": clip_limit,
        "clahe_tile_grid": list(tile),
        "num_gallery": int(len(gallery_set)),
        "num_probe": int(len(probe_set)),
        "num_classes": int(num_classes),
        **{k: round(float(v), 4) if v == v else None for k, v in m.items()},
    }
    print(json.dumps(result, indent=2))
    out = Path(args.out_json) if args.out_json else Path(args.ckpt).with_name("cross_session_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
