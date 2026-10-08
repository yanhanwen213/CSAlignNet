"""
Miura et al. baseline on UTFVP (max-curvature vein extraction + matching).

严格做法: 静脉密度参数(PCT, 取曲率最高的前PCT%像素)只在train内部标定,
test全程不参与任何选择, 每个划分只测一次.
"""
import os, glob, numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.filters import gaussian
from scipy.ndimage import convolve1d
from skimage.morphology import remove_small_objects
from skimage.transform import resize

# Dataset locations: set via environment or edit here.
#   UTFVP_ROI   -> folder holding per-class ROI images (extractedvein)
#   UTFVP_SPLIT -> parent folder holding the UTFVP_3T1T* split folders
ROI_ROOT = os.environ.get("UTFVP_ROI", r"<path-to-UTFVP_ROI\extractedvein>")
_split_parent = os.environ.get("UTFVP_SPLIT", r"<path-to-splits-parent>")
SPLITS = [os.path.join(_split_parent, d) for d in
          ("UTFVP_3T1T", "UTFVP_3T1T_43", "UTFVP_3T1T_44")]
CANON = (432, 144)  # UTFVP ROI为竖图 (h, w)
SHIFTS = [(dy, dx) for dy in range(-6, 7, 2) for dx in range(-6, 7, 2)]
PCT_CANDIDATES = [10, 15, 20, 25, 30, 35]

# ---------- 1. Miura最大曲率: 返回连续分数图 ----------
def max_curvature_score(gray):
    g = gray.astype(np.float64)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    score = np.zeros_like(g)
    for s in [1.0, 2.0, 3.0]:
        f = gaussian(g, s)
        for axis in [0, 1]:
            d1 = convolve1d(f, [-0.5, 0, 0.5], axis=axis)
            d2 = convolve1d(f, [1, -2, 1], axis=axis)
            k = np.abs(d2) / (1 + d1 ** 2) ** 1.5
            score = np.maximum(score, k * (d2 > 0))
        fr = np.rot90(f)
        d2r = convolve1d(fr, [1, -2, 1], axis=0)
        score = np.maximum(score, np.rot90(np.abs(d2r) * (d2r > 0), 3))
    return score

_score_cache = {}
def score_of(path):
    if path not in _score_cache:
        img = resize(imread(path, as_gray=True), CANON, anti_aliasing=True)
        _score_cache[path] = max_curvature_score(img)
    return _score_cache[path]

def vein_at(path, pct):
    smap = score_of(path)
    thr = np.percentile(smap, 100.0 - pct)
    return remove_small_objects(smap > thr, min_size=30).astype(np.float64)

def shift_zero(q, dy, dx):
    h, w = q.shape
    out = np.zeros_like(q)
    out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] = \
        q[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

# ---------- 2. 划分读取 (ROI取自ROI_ROOT, 划分目录只定归属) ----------
def split_classes(sdir):
    root = os.path.join(sdir, "train")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

def paths_of(sdir, part, c):
    fs = sorted(glob.glob(os.path.join(sdir, part, c, "*")))
    return [os.path.join(ROI_ROOT, c, os.path.basename(f)) for f in fs]

# ---------- 3. 在train内部标定PCT (留下每个模板自身不计, 模拟真实检索) ----------
def calibrate_pct(sdir):
    classes = split_classes(sdir)
    best_pct, best_r = None, -1
    for pct in PCT_CANDIDATES:
        Q, cls_idx = [], []
        for ci, c in enumerate(classes):
            for p in paths_of(sdir, "train", c):
                Q.append(vein_at(p, pct).ravel()); cls_idx.append(ci)
        Q = np.array(Q); cls_idx = np.array(cls_idx)
        Q = Q - Q.mean(axis=1, keepdims=True)
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        S = Q @ Q.T
        np.fill_diagonal(S, -np.inf)              # 查询不计自身
        same = cls_idx[:, None] == cls_idx[None, :]
        genuine = np.where(same, S, -np.inf).max(axis=1)
        impostor = np.where(~same, S, -np.inf).max(axis=1)
        r = np.mean(genuine > impostor)
        print(f"  pct={pct}: train_rank1={r*100:.2f}%")
        if r > best_r:
            best_r, best_pct = r, pct
    return best_pct

# ---------- 4. test评测 (多模板 + 位移容忍) ----------
def eer_from_scores(scores, rowcls, gen):
    pos = np.array([scores[i][rowcls == g].max() for i, g in enumerate(gen)])
    neg = np.concatenate([scores[i][rowcls != g] for i, g in enumerate(gen)])
    thrs = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 2000)
    far = np.array([(neg >= t).mean() for t in thrs])
    frr = np.array([(pos < t).mean() for t in thrs])
    i = int(np.argmin(np.abs(far - frr)))
    return (far[i] + frr[i]) / 2

def run_split(sdir, pct):
    classes = split_classes(sdir)
    assert len(classes) == 360, sdir
    tmpl, rowcls = [], []
    for ci, c in enumerate(classes):
        for p in paths_of(sdir, "train", c):
            t = vein_at(p, pct)
            tmpl.append((t - t.mean()).ravel()); rowcls.append(ci)
    G = np.stack(tmpl)
    G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    rowcls = np.array(rowcls)

    queries = [(ci, p) for ci, c in enumerate(classes) for p in paths_of(sdir, "test", c)]
    top1, scores, gen, fails = 0, [], [], []
    for ci, p in tqdm(queries, desc=f"probe {os.path.basename(sdir)}", leave=False):
        q0 = vein_at(p, pct)
        best = np.full(len(tmpl), -np.inf)
        for dy, dx in SHIFTS:
            qs = shift_zero(q0, dy, dx).ravel()
            qs = qs - qs.mean(); qs /= (np.linalg.norm(qs) + 1e-8)
            np.maximum(best, G @ qs, out=best)
        pred = rowcls[best.argmax()]
        if pred == ci:
            top1 += 1
        else:
            fails.append(f"{classes[ci]}\t{classes[pred]}\t{p}")
        scores.append(best); gen.append(ci)
    return top1 / len(queries), eer_from_scores(scores, rowcls, gen), fails

def main():
    print("calibrating PCT on train (test untouched) ...")
    pct = calibrate_pct(SPLITS[0])
    print(f"selected PCT={pct}%")
    r1s, eers = [], []
    for sdir in SPLITS:
        r1, eer, fails = run_split(sdir, pct)
        r1s.append(r1 * 100); eers.append(eer * 100)
        print(f"{os.path.basename(sdir)}: Rank-1={r1*100:.2f}%  EER={eer*100:.2f}%  fails={len(fails)}")
        with open(f"miura_utfvp_failures_{os.path.basename(sdir)}.txt", "w") as f:
            f.write("\n".join(fails))
    print(f"Miura UTFVP (PCT={pct}%, train-calibrated): "
          f"Rank-1={np.mean(r1s):.2f}±{np.std(r1s):.2f}%  "
          f"EER={np.mean(eers):.2f}±{np.std(eers):.2f}%")

if __name__ == "__main__":
    main()
