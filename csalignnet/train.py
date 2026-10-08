"""
Train CSAlignNet (AlignmentHead + Warper + LRFVNet + SessionInvariantHead).

Protocols:
  s1_only (default, main / strict):
    Train on Session-1 only (S2 never used in training).
    Eval: Gallery=S1 → Probe=S2 (Rank-1 / EER on session embeddings).

  joint (ablation / upper bound):
    Train with paired S1+S2 batches (previous behavior).

Example (strict FV-USM + pseudo-session + optional domain-adv):
  python train.py ^
    --session1_root .../1st_session/extractedvein ^
    --session2_root .../2nd_session/extractedvein ^
    --train_protocol s1_only --epochs 120 --batch_size 64 ^
    --lam_sess 1.5 --sess_temperature 0.05 --domain_adv --lam_dom 0.1

CLAHE fine-tune from best pseudo-only ckpt:
  python train.py ^
    --session1_root .../1st_session/extractedvein ^
    --session2_root .../2nd_session/extractedvein ^
    --train_protocol s1_only --clahe --no-domain_adv ^
    --resume .../cs_align_s1_only_pseudo_only/cs_align_best.pth ^
    --epochs 40 --lr 5e-5 --run_name s1_only_pseudo_clahe_ft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# External LRFVNet backbone: <repo-parent>/LRFVNet (or set LRFVNET_PATH).
_LRFV = Path(os.environ.get("LRFVNET_PATH", str(ROOT.parent.parent / "LRFVNet")))
if _LRFV.is_dir() and str(_LRFV) not in sys.path:
    sys.path.insert(0, str(_LRFV))

from logger import EpochLogger
from pidanet.cs_dataset import (
    CrossSessionPoseDataset,
    DualSessionIndex,
    Session1OnlyPoseDataset,
    SessionImageDataset,
)
from pidanet.cs_losses import CSAlignLoss
from pidanet.cs_metrics import evaluate_cross_session
from pidanet.cs_model import CSAlignNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _logits_for_acc(model, out, labels=None):
    if getattr(model, "use_angular", False) and getattr(model, "angular_head", None) is not None:
        try:
            return model.angular_logits(out["embedding"], labels)
        except Exception:
            return out["logits"]
    return out["logits"]


def train_one_epoch(model, loader, criterion, optimizer, device, use_sam=False):
    model.train()
    meters = defaultdict(float)
    n = 0
    for x_a, x_b, x_pose, x_pos, x_neg, labels in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        x_pose = x_pose.to(device)
        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        labels = labels.to(device)

        if not use_sam:
            out_a = model(x_a, return_aux=True)
            out_b = model(x_b, return_aux=True)
            out_pose = model(x_pose, return_aux=True)
            out_p = model(x_pos, return_aux=True)
            out_n = model(x_neg, return_aux=True)
            loss, stats = criterion(out_pose, labels, out_s1=out_a, out_s2=out_b, out_p=out_p, out_n=out_n, out_clean=out_a, model=model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # SAM: first forward/backward -> perturb -> second forward/backward -> update
            out_a = model(x_a, return_aux=True)
            out_b = model(x_b, return_aux=True)
            out_pose = model(x_pose, return_aux=True)
            out_p = model(x_pos, return_aux=True)
            out_n = model(x_neg, return_aux=True)
            loss, stats = criterion(out_pose, labels, out_s1=out_a, out_s2=out_b, out_p=out_p, out_n=out_n, out_clean=out_a, model=model)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # second forward at perturbed weights (same batch, re-use augmented tensors)
            out_a2 = model(x_a, return_aux=True)
            out_b2 = model(x_b, return_aux=True)
            out_pose2 = model(x_pose, return_aux=True)
            out_p2 = model(x_pos, return_aux=True)
            out_n2 = model(x_neg, return_aux=True)
            loss2, _ = criterion(out_pose2, labels, out_s1=out_a2, out_s2=out_b2, out_p=out_p2, out_n=out_n2, out_clean=out_a2, model=model)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

        bs = labels.size(0)
        n += bs
        for k, v in stats.items():
            meters[k] += v * bs
        # acc must use the same logits that were optimized (angular when enabled)
        logits_acc = _logits_for_acc(model, out_a, labels)
        meters["acc"] += (logits_acc.argmax(1) == labels).float().sum().item()
    return {k: v / max(n, 1) for k, v in meters.items()}


def _clahe_kwargs(args) -> dict:
    return dict(
        use_clahe=bool(args.clahe),
        clahe_clip_limit=float(args.clahe_clip_limit),
        clahe_tile_grid=tuple(args.clahe_tile_grid),
    )


def build_train_set(args):
    common = dict(
        session1_root=args.session1_root,
        session2_root=args.session2_root,
        rot_range=tuple(args.rot_range),
        shift_range=tuple(args.shift_range),
        **_clahe_kwargs(args),
    )
    if args.train_protocol == "s1_only":
        return Session1OnlyPoseDataset(
            **common,
            prefer_orig=args.prefer_orig,
            pseudo_session=args.pseudo_session,
            pseudo_session_strength=args.pseudo_session_strength,
            fda=args.fda,
            fda_prob=args.fda_prob,
            fda_beta=args.fda_beta,
        )
    return CrossSessionPoseDataset(**common, pose_from=args.pose_from)


def train(args) -> dict:
    device = torch.device(args.device)
    set_seed(args.seed)

    index = DualSessionIndex(
        args.session1_root, args.session2_root, prefer_orig=args.prefer_orig
    )
    num_classes = index.num_classes
    class_to_idx = index.class_to_idx
    clahe_kw = _clahe_kwargs(args)

    train_set = build_train_set(args)
    gallery_set = SessionImageDataset(
        args.session1_root,
        class_names=index.class_names,
        class_to_idx=class_to_idx,
        orig_only=True,
        **clahe_kw,
    )
    probe_set = SessionImageDataset(
        args.session2_root,
        class_names=index.class_names,
        class_to_idx=class_to_idx,
        orig_only=True,
        **clahe_kw,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    probe_loader = DataLoader(
        probe_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    use_domain_adv = bool(args.domain_adv) and args.lam_dom > 0 and bool(args.use_session_head)
    model = CSAlignNet(
        num_classes=num_classes,
        warp_mode=args.warp_mode,
        tps_grid=args.tps_grid,
        t_scale=args.t_scale,
        session_hidden=args.session_hidden,
        session_out=args.session_out,
        use_domain_adv=use_domain_adv,
        use_alignment=args.use_alignment,
        use_session_head=args.use_session_head,
        style_aug=args.style_aug,
        style_p=args.style_p,
        style_alpha=args.style_alpha,
        use_angular=args.angular,
        angular_type=args.angular_type,
        angular_s=args.angular_s,
        angular_m=args.angular_m,
    ).to(device)
    criterion = CSAlignLoss(
        lam_tri=args.lam_tri,
        lam_sess=args.lam_sess,
        lam_reg=args.lam_reg,
        lam_syn=args.lam_syn,
        lam_dom=args.lam_dom if use_domain_adv else 0.0,
        w_pose=args.w_pose,
        w_s1=args.w_s1,
        w_s2=args.w_s2,
        sess_temperature=args.sess_temperature,
        grl_lambda=args.grl_lambda,
    )
    base_opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_sam = bool(getattr(args, "sam", False))
    if use_sam:
        from sam import SAM as SAMOpt

        optimizer = SAMOpt(model.parameters(), base_opt, rho=float(args.sam_rho), adaptive=bool(args.sam_adaptive))
    else:
        optimizer = base_opt
    # scheduler steps on base optimizer's lr; for SAM it is the same param_groups
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=args.epochs, eta_min=args.lr_min)

    resume_from = ""
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ckpt_nc = int(ckpt.get("num_classes", num_classes))
        if ckpt_nc != num_classes:
            raise RuntimeError(
                f"Resume num_classes mismatch: ckpt={ckpt_nc} data={num_classes}"
            )
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(
                f"Resume load_state_dict: missing={list(missing)} "
                f"unexpected={list(unexpected)}"
            )
        resume_from = str(resume_path)
        print(
            f"Resumed weights from {resume_path} "
            f"(src_epoch={ckpt.get('epoch')}, "
            f"src_r1_sess={ckpt.get('metrics', {}).get('rank1_session')})"
        )

    run_dir = Path(args.log_root) / f"cs_align_{args.run_name}"
    logger = EpochLogger(run_dir.parent, run_name=run_dir.name)
    logger.save_config(
        {
            **vars(args),
            "num_classes": num_classes,
            "param_m": model.count_parameters() / 1e6,
            "class_names_head": index.class_names[:10],
            "ckpt_metric": "rank1_session",
            "eval_protocol": "gallery=session1, probe=session2",
            "train_images": len(train_set),
            "resume_from": resume_from,
        }
    )
    print(
        f"CSAlignNet | protocol={args.train_protocol} | C={num_classes} "
        f"train_n={len(train_set)} params={model.count_parameters()/1e6:.3f}M | "
        f"log={logger.path()} | ckpt=rank1_session"
    )
    if args.clahe:
        print(
            f"CLAHE: on (clip={args.clahe_clip_limit}, "
            f"tile={tuple(args.clahe_tile_grid)}) — train+eval"
        )
    else:
        print("CLAHE: off")
    if args.train_protocol == "s1_only":
        print("Strict: S2 images are NOT used in training; only for S1→S2 eval.")
        print(
            f"Pseudo-session photometric on x_b: "
            f"{args.pseudo_session} (strength={args.pseudo_session_strength})"
        )
    if use_domain_adv:
        print(
            f"Domain-adv on session emb: on "
            f"(lam_dom={args.lam_dom}, grl_lambda={args.grl_lambda})"
        )
    print(f"Alignment: {'on (AlignmentHead+Warper)' if args.use_alignment else 'OFF (identity passthrough)'}")
    print(
        "Session head: "
        f"{'on (session-invariant projection)' if args.use_session_head else 'OFF (use backbone emb for retrieval)'}"
    )

    sam_flag = bool(getattr(args, "sam", False))
    if sam_flag:
        print(f"SAM: on (rho={args.sam_rho}, adaptive={args.sam_adaptive}) ~2x time/epoch (Foret et al., ICLR 2021)")
    else:
        print("SAM: off")
    if getattr(args, "angular", False):
        print(f"Angular: {args.angular_type} s={args.angular_s} m={args.angular_m} (Deng CVPR19 / Wang CVPR18)")
    else:
        print("Angular: off (plain CE)")

    best_rank1 = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    ckpt_path = logger.path() / "cs_align_best.pth"

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, criterion, optimizer, device, use_sam=sam_flag)
        cs = evaluate_cross_session(
            model, gallery_loader, probe_loader, device, num_classes=num_classes
        )
        scheduler.step()
        train_acc = 100.0 * stats["acc"]
        lr_now = optimizer.param_groups[0]["lr"]
        dom_str = f" dom={stats['dom']:.4f}" if use_domain_adv else ""
        print(
            f"Ep{epoch:03d} loss={stats['loss']:.4f} id={stats['id']:.4f} "
            f"sess={stats['sess']:.4f} tri={stats['tri']:.4f}{dom_str} "
            f"train_acc={train_acc:.2f}% "
            f"probe_acc={cs['probe_cls_acc']:.2f}% "
            f"r1_sess={cs['rank1_session']:.2f}% r1_emb={cs['rank1_embedding']:.2f}% "
            f"eer_sess={cs['eer_session']:.2f}% eer_emb={cs['eer_embedding']:.2f}% "
            f"lr={lr_now:.2e}"
        )
        logger.log_epoch(
            epoch,
            {
                "train_loss": round(stats["loss"], 6),
                "train_id": round(stats["id"], 6),
                "train_sess": round(stats["sess"], 6),
                "train_tri": round(stats["tri"], 6),
                "train_reg": round(stats["reg"], 6),
                "train_syn": round(stats["syn"], 6),
                "train_dom": round(stats["dom"], 6),
                "train_acc": round(train_acc, 4),
                "probe_cls_acc": round(cs["probe_cls_acc"], 4),
                "probe_cls_f1": round(cs["probe_cls_f1"], 4),
                "rank1_session": round(cs["rank1_session"], 4),
                "rank1_embedding": round(cs["rank1_embedding"], 4),
                "eer_session": round(cs["eer_session"], 4),
                "eer_embedding": round(cs["eer_embedding"], 4),
                "lr": round(lr_now, 8),
            },
        )
        if cs["rank1_session"] > best_rank1:
            best_rank1 = float(cs["rank1_session"])
            best_epoch = epoch
            best_metrics = dict(cs)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "class_names": index.class_names,
                    "metrics": cs,
                    "train_protocol": args.train_protocol,
                },
                ckpt_path,
            )

    summary = {
        "train_protocol": args.train_protocol,
        "best_epoch": best_epoch,
        "best_rank1_session": best_rank1,
        "best_metrics": best_metrics,
        "ckpt": str(ckpt_path),
        "num_classes": num_classes,
    }
    with open(logger.path() / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("===== done =====")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train CSAlignNet")
    p.add_argument("--session1_root", type=str, required=True)
    p.add_argument("--session2_root", type=str, required=True)
    p.add_argument(
        "--train_protocol",
        type=str,
        default="s1_only",
        choices=["s1_only", "joint"],
        help="s1_only=strict (train S1 only); joint=train with S1+S2 pairs",
    )
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warp_mode", type=str, default="rt", choices=["rt", "affine", "tps"])
    p.add_argument("--tps_grid", type=int, default=3)
    p.add_argument("--t_scale", type=float, default=0.5, help="Warper translation bound (norm coords)")
    p.add_argument("--lam_tri", type=float, default=0.5)
    p.add_argument("--lam_sess", type=float, default=1.0)
    p.add_argument("--lam_reg", type=float, default=0.01)
    p.add_argument("--lam_syn", type=float, default=0.1)
    p.add_argument("--w_pose", type=float, default=1.0)
    p.add_argument("--w_s1", type=float, default=1.0)
    p.add_argument("--w_s2", type=float, default=1.0)
    p.add_argument("--sess_temperature", type=float, default=0.07)
    p.add_argument("--rot_range", type=float, nargs=2, default=[-25, 25])
    p.add_argument("--shift_range", type=float, nargs=2, default=[-15, 15])
    p.add_argument(
        "--prefer_orig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, prefer *orig* images; use --no-prefer_orig for posed ROI sets (e.g. data_roi_bbox)",
    )
    p.add_argument(
        "--pseudo_session",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="s1_only: photometric pseudo-session aug on x_b (default on)",
    )
    p.add_argument(
        "--pseudo_session_strength",
        type=float,
        default=1.0,
        help="Strength of photometric pseudo-session aug",
    )
    p.add_argument(
        "--domain_adv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional GRL domain classifier on session embeddings (a=0, b=1)",
    )
    p.add_argument(
        "--lam_dom",
        type=float,
        default=0.1,
        help="Weight for domain-adversarial CE (used when --domain_adv)",
    )
    p.add_argument(
        "--grl_lambda",
        type=float,
        default=1.0,
        help="Gradient reversal scale for domain adversarial training",
    )
    p.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply CLAHE on load for train gallery/probe (must match at eval)",
    )
    p.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=2.0,
        help="OpenCV CLAHE clipLimit",
    )
    p.add_argument(
        "--clahe_tile_grid",
        type=int,
        nargs=2,
        default=[8, 8],
        help="OpenCV CLAHE tileGridSize H W",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Load model weights from ckpt (fine-tune; optimizer fresh)",
    )
    p.add_argument(
        "--angular",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Angular-margin CE (CosFace/ArcFace) on backbone emb (training-only, zero infer params)",
    )
    p.add_argument("--angular_type", type=str, default="cosface", choices=["cosface", "arcface"])
    p.add_argument("--angular_s", type=float, default=64.0, help="Feature scale s")
    p.add_argument("--angular_m", type=float, default=0.35, help="Margin m (radians for arcface, cosine for cosface)")
    p.add_argument(
        "--fda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="s1_only: Fourier Domain Adaptation low-freq swap on x_b (Yang & Soatto, CVPR 2020)",
    )
    p.add_argument("--fda_prob", type=float, default=0.5)
    p.add_argument("--fda_beta", type=float, default=0.05)
    p.add_argument(
        "--style_aug",
        type=str,
        default="none",
        choices=["none", "mixstyle", "dsu", "padain"],
        help="Feature-statistics style aug after conv1/conv2 (Zhou ICLR21 / Li ICLR22)",
    )
    p.add_argument("--style_p", type=float, default=0.5)
    p.add_argument("--style_alpha", type=float, default=0.1, help="Beta alpha for MixStyle")
    p.add_argument(
        "--sam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sharpness-Aware Minimization wrapper (Foret et al., ICLR 2021); ~2x time/epoch",
    )
    p.add_argument("--sam_rho", type=float, default=0.05, help="SAM perturbation radius")
    p.add_argument("--sam_adaptive", action=argparse.BooleanOptionalAction, default=False, help="ASAM adaptive mode")
    p.add_argument("--pose_from", type=str, default="s1", choices=["s1", "any"])
    p.add_argument(
        "--use_alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AlignmentHead+Warper (default on); --no-use_alignment removes the pose head (identity passthrough)",
    )
    p.add_argument(
        "--use_session_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SessionInvariantHead (default on); --no-use_session_head uses backbone emb for retrieval",
    )
    p.add_argument("--session_hidden", type=int, default=256)
    p.add_argument("--session_out", type=int, default=128)
    p.add_argument("--log_root", type=str, default=str(ROOT / "log" / "cs_align"))
    p.add_argument("--run_name", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run_name:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.train_protocol}_{stamp}"
    train(args)


if __name__ == "__main__":
    main()
