"""
Error attribution on FV-USM strict protocol (three seeds):
把 S2 探针的失败样本归因为 姿态 / 光照 / 本质难分 三类.

输入: 三个 FT30 严格协议 ckpt (seed 42/43/44).
输出: 各种子占比 + 三种子 mean±std + 失败清单(seed44).

复现:
  python error_attribution.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

REPO = Path(__file__).resolve().parents[1]  # repository root
_CSALIGNNET = str(REPO / "csalignnet")
if _CSALIGNNET not in sys.path:
    sys.path.insert(0, _CSALIGNNET)
# LRFVNet backbone is an external dependency, expected next to this repo:
#   <parent>/LRFVNet   (or set LRFVNET_PATH)
_LRFV = Path(os.environ.get("LRFVNET_PATH", str(REPO.parent / "LRFVNet")))
if str(_LRFV) not in sys.path:
    sys.path.insert(0, str(_LRFV))

from pidanet.cs_model import CSAlignNet          # noqa: E402
from pidanet.cs_dataset import (                  # noqa: E402
    DualSessionIndex, SessionImageDataset, build_default_transform)

# Directory holding the three FT30 run folders (override with --ckpt_dir).
CKPT_DIR = os.environ.get("CSALIGN_CKPT_DIR", str(REPO / "checkpoints" / "cs_align"))
RUNS = {
    42: "cs_align_s1_only_mixstyle_cosface_ft30",
    43: "cs_align_s1_only_mixstyle_cosface_ft30_s43",
    44: "cs_align_s1_only_mixstyle_cosface_ft30_s44",
}
OUT = str(REPO / "evaluation" / "attribution_3seeds.json")

ROT_SCALE = math.pi / 4
T_SCALE = 0.5
IMAGE = 112
PHOTO_SIGMA = 2.0         # 光照偏离阈值(σ)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_model(seed, device, ckpt_dir):
    run = RUNS[seed]
    cfg = json.load(open(os.path.join(ckpt_dir, run, "config.json"), encoding="utf-8"))
    model = CSAlignNet(
        num_classes=cfg["num_classes"], warp_mode=cfg.get("warp_mode", "rt"),
        tps_grid=cfg.get("tps_grid", 3), t_scale=cfg.get("t_scale", 0.5),
        session_hidden=cfg.get("session_hidden", 256),
        session_out=cfg.get("session_out", 128),
        use_alignment=True, use_session_head=True,
        style_aug="none",  # 推理期不启用mixstyle
        use_angular=cfg.get("angular", True),
        angular_type=cfg.get("angular_type", "cosface"),
        angular_s=cfg.get("angular_s", 64.0), angular_m=cfg.get("angular_m", 0.35),
    )
    ckpt = torch.load(os.path.join(ckpt_dir, run, "cs_align_best.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval(), cfg


@torch.no_grad()
def extract(model, loader, device):
    embs, y_all = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        aux = model(x, return_aux=True)
        embs.append(aux["session_embedding"].cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(embs), np.concatenate(y_all)


def norm_template(emb, y):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    classes = sorted(set(y.tolist()))
    templ = np.stack([e[y == c].mean(0) for c in classes])
    templ /= np.linalg.norm(templ, axis=1, keepdims=True) + 1e-12
    return templ, np.asarray(classes)


def warped_theta(theta_rad, tx, ty):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return torch.tensor([[c, -s, tx], [s, c, ty]], dtype=torch.float32)


def img_stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return float(a.mean()), float(a.std())


def run_seed(seed, device, ckpt_dir, keep_detail=False):
    """单一种子的完整归因, 返回 summary 字典(keep_detail 时附 detail 列表)。"""
    set_seed(seed)
    model, cfg = load_model(seed, device, ckpt_dir)
    s1_root, s2_root = cfg["session1_root"], cfg["session2_root"]

    index = DualSessionIndex(s1_root, s2_root, prefer_orig=True)
    gallery_set = SessionImageDataset(s1_root, class_names=index.class_names,
                                      class_to_idx=index.class_to_idx,
                                      image_size=IMAGE, orig_only=True)
    probe_set = SessionImageDataset(s2_root, class_names=index.class_names,
                                    class_to_idx=index.class_to_idx,
                                    image_size=IMAGE, orig_only=True)
    gl = DataLoader(gallery_set, batch_size=64, shuffle=False)
    pl = DataLoader(probe_set, batch_size=64, shuffle=False)

    g_emb, g_y = extract(model, gl, device)
    p_emb, p_y = extract(model, pl, device)

    templ, cls = norm_template(g_emb, g_y)
    pe = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-12)
    scores = pe @ templ.T
    pred = cls[scores.argmax(1)]
    rank1 = float((pred == p_y).mean() * 100)

    fail_idx = np.where(pred != p_y)[0]
    n_fail = len(fail_idx)

    # 类模板的光照统计 (S1内该类各图 mean/std)
    paths_g = [str(s[0]) for s in gallery_set.samples]
    stats_g = np.array([img_stats(p) for p in paths_g])
    cls_stats = {}
    for c in cls:
        m = g_y == c
        cls_stats[int(c)] = (stats_g[m, 0].mean(), stats_g[m, 1].mean())

    paths_p = [str(s[0]) for s in probe_set.samples]
    tf = build_default_transform(IMAGE)

    # 全局光照阈值: 所有探针与其真类模板的 (亮度差, 对比度差) 分布 mean+2std
    gaps_mean, gaps_std = [], []
    for i in range(len(p_y)):
        mean_i, std_i = img_stats(paths_p[i])
        cm, sm = cls_stats[int(p_y[i])]
        gaps_mean.append(abs(mean_i - cm)); gaps_std.append(abs(std_i - sm))
    gm = np.asarray(gaps_mean); gsd = np.asarray(gaps_std)
    thr_mean = gm.mean() + PHOTO_SIGMA * gm.std()
    thr_std = gsd.mean() + PHOTO_SIGMA * gsd.std()

    def _corrective_probe(path, cls_target):
        """纠正性配准网格搜索: 对原始图施加旋转/平移, 过'去姿态头'分支反查。
        返回 True 若任一合理摆正能匹配到真类。"""
        pil = Image.open(path).convert("L")
        x0 = tf(pil)
        thetas = [-12, -6, 0, 6, 12]          # 度
        shifts = [-8, -4, 0, 4, 8]            # 像素(112系)
        batch, metas = [], []
        for ang in thetas:
            for dy in shifts:
                for dx in shifts:
                    th = warped_theta(math.radians(ang), dx * 2 / IMAGE, dy * 2 / IMAGE)
                    grid = F.affine_grid(th.unsqueeze(0), size=x0.unsqueeze(0).size(),
                                         align_corners=True)
                    xw = F.grid_sample(x0.unsqueeze(0), grid, mode="bilinear",
                                       padding_mode="border", align_corners=True)
                    batch.append(xw[0]); metas.append((ang, dx, dy))
        xb = torch.stack(batch).to(device)
        prev = model.use_alignment
        model.use_alignment = False
        with torch.no_grad():
            aux = model(xb, return_aux=True)
        model.use_alignment = prev
        e = aux["session_embedding"].cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        sims = e @ templ.T
        preds = cls[sims.argmax(1)]
        hit = np.where(preds == cls_target)[0]
        if len(hit):
            tidx = int(np.where(cls == cls_target)[0][0])
            k = hit[sims[hit, tidx].argmax()]
            return True, metas[k]
        return False, None

    n_pose, n_photo, n_intr, recovered, detail = 0, 0, 0, 0, []
    for i in tqdm_wrap(fail_idx, desc=f"seed{seed}"):
        y_true = int(p_y[i]); y_pred = int(pred[i])
        rec, meta = _corrective_probe(paths_p[i], y_true)
        mean_i, std_i = img_stats(paths_p[i])
        cm, sm = cls_stats[y_true]
        d_mean = abs(mean_i - cm); d_contrast = abs(std_i - sm)

        if rec:
            n_pose += 1; recovered += 1
        elif (d_mean > thr_mean) or (d_contrast > thr_std):
            n_photo += 1
        else:
            n_intr += 1
        if keep_detail:
            detail.append({"probe": Path(paths_p[i]).name, "true": int(y_true),
                           "pred": int(y_pred), "recovered": bool(rec),
                           "fix": meta, "d_mean": round(float(d_mean), 4),
                           "d_contrast": round(float(d_contrast), 4)})
    res = {
        "seed": seed, "rank1": rank1, "n_probe": int(len(p_y)), "n_fail": int(n_fail),
        "pose": n_pose, "photometric": n_photo, "intrinsic": n_intr,
        "pct_pose": 100.0 * n_pose / max(1, n_fail),
        "pct_photometric": 100.0 * n_photo / max(1, n_fail),
        "pct_intrinsic": 100.0 * n_intr / max(1, n_fail),
    }
    if keep_detail:
        res["detail"] = detail
    return res


def tqdm_wrap(it, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, leave=False)
    except Exception:
        return it


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return round(float(a.mean()), 2), round(float(a.std()), 2)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="3-seed residual-error attribution")
    ap.add_argument("--ckpt_dir", default=CKPT_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    ckpt_dir, out = args.ckpt_dir, args.out
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    per_seed = []
    for seed in [42, 43, 44]:
        print(f"===== seed {seed} =====")
        r = run_seed(seed, device, ckpt_dir, keep_detail=(seed == 44))
        per_seed.append(r)
        print(f"seed{seed}: Rank-1={r['rank1']:.2f}% n_fail={r['n_fail']} "
              f"pose={r['pct_pose']:.1f}% photo={r['pct_photometric']:.1f}% "
              f"intr={r['pct_intrinsic']:.1f}%")
    agg = {
        "rank1": mean_std([r["rank1"] for r in per_seed]),
        "n_fail": mean_std([r["n_fail"] for r in per_seed]),
        "pct_pose": mean_std([r["pct_pose"] for r in per_seed]),
        "pct_photometric": mean_std([r["pct_photometric"] for r in per_seed]),
        "pct_intrinsic": mean_std([r["pct_intrinsic"] for r in per_seed]),
    }
    print("===== 3-seed aggregate (mean, std) =====")
    print(json.dumps(agg, indent=2))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_seed": per_seed, "aggregate": agg}, f,
                  ensure_ascii=False, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
