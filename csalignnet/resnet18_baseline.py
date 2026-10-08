"""公平对比 ResNet18 (torchvision实现, 非timm): 112x112x1 from-scratch, 与其它基线同协议/指标.
首层conv1由3ch改为1ch (权重按通道均值折叠); 评测: avgpool后特征L2 + 模板均值 + cosine最近邻 + 标准1:1 EER.
"""
import argparse, os, random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from torch.utils.data import Dataset


def fv_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112, 112)),
        transforms.RandomAffine(degrees=25, translate=(0.134, 0.134), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def fv_test_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

class FVFolder(Dataset):
    IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.pgm')
    _SPLIT_ALIAS = {'S1': '1st_session/extractedvein', 'S2': '2nd_session/extractedvein',
                    'train': 'train', 'test': 'test'}
    def __init__(self, root, split='train', transform=None):
        self.transform = transform or fv_test_transform()
        d = os.path.join(root, split)
        if not os.path.isdir(d) and split in self._SPLIT_ALIAS:
            d = os.path.join(root, self._SPLIT_ALIAS[split])
        assert os.path.isdir(d), f'not found: {d}'
        classes = sorted([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))])
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for c in classes:
            cd = os.path.join(d, c)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(self.IMG_EXTS):
                    self.samples.append((os.path.join(cd, f), self.cls2idx[c]))
        assert len(self.samples) > 0
        self.nb_classes = len(classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p)), y


def build_resnet18_1ch(num_classes):
    net = models.resnet18(weights=None)
    w = net.conv1.weight.data  # [64,3,7,7]
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = w.mean(dim=1, keepdim=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet_embed(net, x):
    x = net.conv1(x); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
    x = net.layer1(x); x = net.layer2(x); x = net.layer3(x); x = net.layer4(x)
    x = net.avgpool(x)
    return F.normalize(torch.flatten(x, 1), dim=1)


@torch.no_grad()
def retrieval_eval(net, gallery_loader, probe_loader, device):
    net.eval()
    def embed(loader):
        feats, labels = [], []
        for x, y in loader:
            feats.append(resnet_embed(net, x.to(device)).cpu()); labels.append(y)
        return torch.cat(feats), torch.cat(labels)
    g_f, g_y = embed(gallery_loader)
    p_f, p_y = embed(probe_loader)
    classes = g_y.unique().tolist()
    tmpl = torch.stack([F.normalize(g_f[g_y == c].mean(0, keepdim=True), dim=1).squeeze(0) for c in classes])
    sim = p_f @ tmpl.T
    pred = torch.tensor([classes[i] for i in sim.argmax(1).tolist()])
    rank1 = (pred == p_y).float().mean().item() * 100
    c2row = {c: i for i, c in enumerate(classes)}
    rows = torch.tensor([c2row[int(c)] for c in p_y.tolist()])
    genuine = sim[torch.arange(len(sim)), rows]
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(sim)), rows] = False
    impostor = sim[mask]
    ths = torch.linspace(-1, 1, 2001)
    far = [(impostor >= t).float().mean().item() for t in ths]
    frr = [(genuine < t).float().mean().item() for t in ths]
    i = int(np.argmin([abs(a - b) for a, b in zip(far, frr)]))
    return rank1, (far[i] + frr[i]) / 2 * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--epochs', type=int, default=160)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min-lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--gallery-split', default='train')
    ap.add_argument('--probe-split', default='test')
    ap.add_argument('--out', default='./fv_out_resnet18')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = FVFolder(args.data_root, args.train_split, fv_train_transform())
    gal_set = FVFolder(args.data_root, args.gallery_split, fv_test_transform())
    prb_set = FVFolder(args.data_root, args.probe_split, fv_test_transform())
    nc = train_set.nb_classes
    print(f'resnet18-1ch classes={nc} trainN={len(train_set)} galN={len(gal_set)} prbN={len(prb_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    gal_loader = DataLoader(gal_set, batch_size=256, shuffle=False, num_workers=0)
    prb_loader = DataLoader(prb_set, batch_size=256, shuffle=False, num_workers=0)
    net = build_resnet18_1ch(nc).to(device)
    print(f'params={sum(p.numel() for p in net.parameters())/1e6:.2f}M')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    ce = nn.CrossEntropyLoss()
    os.makedirs(args.out, exist_ok=True)
    best = (0, 100)
    for ep in range(1, args.epochs + 1):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(net(x), y)
            loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == args.epochs:
            r1, eer = retrieval_eval(net, gal_loader, prb_loader, device)
            print(f'ep{ep}: Rank-1={r1:.2f} EER={eer:.2f}', flush=True)
            if r1 > best[0]:
                best = (r1, eer)
                torch.save(net.state_dict(), os.path.join(args.out, 'best.pth'))
    torch.save(net.state_dict(), os.path.join(args.out, 'last.pth'))
    print(f'BEST Rank-1={best[0]:.2f} EER={best[1]:.2f}')


if __name__ == '__main__':
    main()
