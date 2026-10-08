"""
Shaheed et al. baseline on UTFVP: ImageNet-pretrained transfer + fine-tune.

与主实验相同的3个3-train/1-test划分. 每个划分: 在train(3张/类)上微调,
在test(1张/类)上评测, 报告 mean±std. 使用外部ImageNet权重(表注需说明).

用法:
  python shaheed_utfvp.py --backbone resnet50 --epochs 40
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shaheed_baseline import build_model, DEFAULT_MEAN, DEFAULT_STD  # noqa: E402

# Dataset locations: set via environment or edit here.
#   UTFVP_ROI   -> folder holding per-class ROI images (extractedvein)
#   UTFVP_SPLIT -> parent folder holding the UTFVP_3T1T* split folders
ROI_ROOT = os.environ.get("UTFVP_ROI", r"<path-to-UTFVP_ROI\extractedvein>")
_split_parent = os.environ.get("UTFVP_SPLIT", r"<path-to-splits-parent>")
SPLITS = [os.path.join(_split_parent, d) for d in
          ("UTFVP_3T1T", "UTFVP_3T1T_43", "UTFVP_3T1T_44")]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_classes(sdir):
    root = os.path.join(sdir, "train")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def paths_of(sdir, part, c):
    fs = sorted(glob.glob(os.path.join(sdir, part, c, "*")))
    return [os.path.join(ROI_ROOT, c, os.path.basename(f)) for f in fs]


class SplitDataset(Dataset):
    def __init__(self, sdir, classes, part, image_size, train):
        self.items = []
        self.image_size = image_size
        for ci, c in enumerate(classes):
            for p in paths_of(sdir, part, c):
                self.items.append((p, ci))
        if train:
            self.tf = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08),
                                        scale=(0.95, 1.05)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([DEFAULT_MEAN], [DEFAULT_STD]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([DEFAULT_MEAN], [DEFAULT_STD]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("L")
        return self.tf(img), y


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x, return_aux=True)
        embs.append(out["embedding"].cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(embs), np.concatenate(ys)


def evaluate(model, gl, pl, device):
    g, gy = extract(model, gl, device)
    p, py = extract(model, pl, device)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    templs = np.stack([g[gy == c].mean(0) for c in sorted(set(gy.tolist()))])
    templs /= np.linalg.norm(templs, axis=1, keepdims=True) + 1e-12
    tt = np.asarray(sorted(set(gy.tolist())))
    nn = (p @ templs.T).argmax(1)
    rank1 = float((tt[nn] == py).mean() * 100)
    genuine, impostor = [], []
    for i, yi in enumerate(py):
        s = p[i] @ templs.T
        genuine.append(float(s[tt == yi][0]))
        impostor.extend(s[tt != yi].tolist())
    genuine = np.asarray(genuine); impostor = np.asarray(impostor)
    thrs = np.linspace(min(impostor.min(), genuine.min()),
                       max(impostor.max(), genuine.max()), 2000)
    far = np.array([(impostor >= t).mean() for t in thrs])
    frr = np.array([(genuine < t).mean() for t in thrs])
    eer = (far + frr)[np.argmin(np.abs(far - frr))] / 2 * 100
    return rank1, eer


def train_split(sdir, args, device):
    classes = split_classes(sdir)
    assert len(classes) == 360, sdir
    train_set = SplitDataset(sdir, classes, "train", args.image_size, True)
    gallery_set = SplitDataset(sdir, classes, "train", args.image_size, False)
    probe_set = SplitDataset(sdir, classes, "test", args.image_size, False)
    tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=False)
    gl = DataLoader(gallery_set, batch_size=args.batch_size, num_workers=args.num_workers)
    pl = DataLoader(probe_set, batch_size=args.batch_size, num_workers=args.num_workers)

    set_seed(args.seed)
    model, _, _ = build_model(args.backbone, len(classes), device)
    model = model.to(device)
    opt = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.backbone_lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_r1, best_eer = -1.0, -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        r1, eer = evaluate(model, gl, pl, device)
        if r1 > best_r1:
            best_r1, best_eer = r1, eer
    print(f"{os.path.basename(sdir)}: best Rank-1={best_r1:.2f}%  EER={best_eer:.2f}%")
    return best_r1, best_eer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    r1s, eers = [], []
    for sdir in SPLITS:
        r1, eer = train_split(sdir, args, args.device)
        r1s.append(r1); eers.append(eer)
    print(f"Shaheed UTFVP ({args.backbone}): "
          f"Rank-1={np.mean(r1s):.2f}±{np.std(r1s):.2f}%  "
          f"EER={np.mean(eers):.2f}±{np.std(eers):.2f}%")


if __name__ == "__main__":
    main()
