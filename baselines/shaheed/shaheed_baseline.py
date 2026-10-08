"""
Shaheed et al. baseline: ImageNet-pretrained transfer + fine-tune.

协议: 严格跨会话. 只在 Session-1 上微调预训练骨干, Gallery=S1(类均值模板),
Probe=S2, 全程 S2 不参与训练/选模. 与 CSAlignNet 的严格协议完全一致.

注意: 使用 ImageNet 预训练权重 = 引入外部数据, 表中需加脚注说明.

用法:
  python shaheed_baseline.py --s1_root .../1st_session/extractedvein ^
      --s2_root .../2nd_session/extractedvein --backbone resnet50 --epochs 40

  UTFVP(3个3-train/1-test划分)见 shaheed_utfvp.py.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]  # repository root
_CSALIGNNET = str(REPO / "csalignnet")
if _CSALIGNNET not in sys.path:
    sys.path.insert(0, _CSALIGNNET)

from pidanet.cs_dataset import DualSessionIndex, SessionImageDataset  # noqa: E402
from pidanet.cs_metrics import rank1_identification, probe_gallery_eer  # noqa: E402

DEFAULT_MEAN, DEFAULT_STD = 0.5, 0.5


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class TrainDataset(Dataset):
    """S1训练集 + 轻量刚性增强(与主实验同量级)."""

    def __init__(self, root, class_names, class_to_idx, image_size, train=True):
        self.base = SessionImageDataset(
            root, class_names=class_names, class_to_idx=class_to_idx,
            image_size=image_size, orig_only=True)
        if train:
            self.tf = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08),
                                        scale=(0.95, 1.05)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([DEFAULT_MEAN], [DEFAULT_STD]),
            ])
        else:
            self.tf = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        if self.tf is not None:
            # base返回已是tensor, 反归一化回PIL做增强
            arr = (img * DEFAULT_STD + DEFAULT_MEAN).clamp(0, 1)
            pil = transforms.ToPILImage()(arr[0] if arr.dim() == 3 else arr)
            img = self.tf(pil)
        return img, label


def build_model(backbone: str, num_classes: int, device: str):
    """返回 (model, embed_dim, is_timm)."""
    try:
        import timm
        if backbone in timm.list_models(pretrained=True):
            model = timm.create_model(backbone, pretrained=True, num_classes=0)
            embed_dim = model.num_features
            head = nn.Linear(embed_dim, num_classes)
            net = _TimmWrap(model, head)
            return net, embed_dim, True
    except Exception as e:
        print(f"[timm] unavailable/failed ({e}); falling back to torchvision")
    import torchvision.models as tvm
    model = tvm.__dict__[backbone](weights="DEFAULT")
    if backbone.startswith("resnet"):
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported torchvision backbone: {backbone}")
    head = nn.Linear(embed_dim, num_classes)
    return _TVWrap(model, head), embed_dim, False


class _TimmWrap(nn.Module):
    def __init__(self, backbone, head):
        super().__init__(); self.backbone = backbone; self.head = head

    def forward(self, x, return_aux=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        emb = self.backbone(x)
        logits = self.head(emb)
        if return_aux:
            return {"embedding": emb, "logits": logits, "session_embedding": emb}
        return logits


class _TVWrap(nn.Module):
    def __init__(self, backbone, head):
        super().__init__(); self.backbone = backbone; self.head = head

    def forward(self, x, return_aux=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        emb = self.backbone(x)
        logits = self.head(emb)
        if return_aux:
            return {"embedding": emb, "logits": logits, "session_embedding": emb}
        return logits


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x, return_aux=True)
        embs.append(out["embedding"].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(embs), np.concatenate(ys)


def evaluate(model, gallery_loader, probe_loader, device):
    g, gy = extract(model, gallery_loader, device)
    p, py = extract(model, probe_loader, device)
    return rank1_identification(g, gy, p, py), probe_gallery_eer(g, gy, p, py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_root", required=True)
    ap.add_argument("--s2_root", required=True)
    ap.add_argument("--backbone", default="resnet50",
                    help="timm名(如 xception)或 torchvision resnet18/34/50")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_seed(args.seed)

    index = DualSessionIndex(args.s1_root, args.s2_root, prefer_orig=True)
    C = index.num_classes
    print(f"classes={C} backbone={args.backbone} size={args.image_size}")

    train_set = TrainDataset(args.s1_root, index.class_names, index.class_to_idx,
                             args.image_size, train=True)
    gallery_set = SessionImageDataset(args.s1_root, class_names=index.class_names,
                                      class_to_idx=index.class_to_idx,
                                      image_size=args.image_size, orig_only=True)
    probe_set = SessionImageDataset(args.s2_root, class_names=index.class_names,
                                    class_to_idx=index.class_to_idx,
                                    image_size=args.image_size, orig_only=True)
    tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    gl = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    pl = DataLoader(probe_set, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    model, embed_dim, _ = build_model(args.backbone, C, args.device)
    model = model.to(args.device)
    opt = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n],
         "lr": args.backbone_lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": args.lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_r1, best = -1.0, {}

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n, correct = 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        r1, eer = evaluate(model, gl, pl, args.device)
        print(f"Ep{ep:03d} loss={tot/n:.4f} train_acc={100*correct/n:.2f}% "
              f"r1(S1->S2)={r1:.2f}% EER={eer:.2f}%")
        if r1 > best_r1:
            best_r1, best = r1, {"epoch": ep, "rank1": r1, "eer": eer}

    print("===== best =====")
    print(json.dumps({"backbone": args.backbone, "image_size": args.image_size,
                      **best}, indent=2))


if __name__ == "__main__":
    main()
