"""
Asaari et al. baseline on UTFVP: BLPOC + WCCD score-level fusion.
Asaari et al., "Fusion of band-limited phase-only correlation and width
centroid contour distance...", ESWA / IEEE TIM 2014.

严格做法:
  * 与主实验相同的3个3-train/1-test划分;
  * 带限比例 KL 与融合权重 w 都只在train内部(留一)标定, test只测一次;
  * 批量FFT加速.
"""
import os, glob, numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.transform import resize

# Dataset locations: set via environment or edit here.
#   UTFVP_ROI   -> folder holding per-class ROI images (extractedvein)
#   UTFVP_SPLIT -> parent folder holding the UTFVP_3T1T* split folders
ROI_ROOT = os.environ.get("UTFVP_ROI", r"<path-to-UTFVP_ROI\extractedvein>")
_split_parent = os.environ.get("UTFVP_SPLIT", r"<path-to-splits-parent>")
SPLITS = [os.path.join(_split_parent, d) for d in
          ("UTFVP_3T1T", "UTFVP_3T1T_43", "UTFVP_3T1T_44")]
CANON = (432, 144)          # UTFVP ROI为竖图 (h, w)
WCCD_SCALE = 0.5
KL_CANDIDATES = [0.02, 0.04, 0.06, 0.08, 0.12]

# ---------- 1. BLPOC (批量) ----------
def band_mask(shape, kl):
    M, N = shape
    m, n = max(1, int(M * kl / 2)), max(1, int(N * kl / 2))
    mask = np.zeros(shape)
    mask[:m, :n] = mask[-m:, :n] = mask[:m, -n:] = mask[-m:, -n:] = 1.0
    return mask

def blpoc_transform(img):
    F = np.fft.fft2(img - img.mean())
    return F / (np.abs(F) + 1e-8)

def blpoc_scores_batch(pq, G, mask, B, MN):
    """pq:(H,W) complex; G:(C,H,W) complex -> (C,) 峰值分数。"""
    R = np.fft.ifft2(pq[None, :, :] * np.conj(G) * mask[None, :, :], axes=(1, 2))
    return np.abs(R).max(axis=(1, 2)) * MN / B

def band_params(shape, kl):
    m = band_mask(shape, kl)
    return m, m.sum(), shape[0] * shape[1]

# ---------- 2. WCCD ----------
def wccd_feature(gray):
    g = gray / (gray.max() + 1e-8)
    m = g > threshold_otsu(g)
    m = remove_small_objects(m, min_size=200)
    lab = label(m)
    if lab.max() == 0:
        return np.zeros(gray.shape[0] * 2)
    cnt = np.bincount(lab.ravel()); cnt[0] = 0
    m = lab == cnt.argmax()
    h, w = m.shape
    has = m.any(axis=1)
    first = np.argmax(m, axis=1)
    last = w - 1 - np.argmax(m[:, ::-1], axis=1)
    center = np.zeros(h); width = np.zeros(h)
    center[has] = (first[has] + last[has]) / 2
    width[has] = last[has] - first[has]
    cx = center[has].mean() if has.any() else 0.0
    mw = (width[has].mean() if has.any() else 1.0) + 1e-8
    return np.concatenate([(center - cx) / mw, width / mw])

def wccd_sim_batch(pw, gw_mat):
    d = np.linalg.norm(gw_mat - pw, axis=1) / len(pw)
    return 1.0 / (1.0 + d / WCCD_SCALE)

# ---------- 3. 数据 ----------
def roi_path(sdir, cls, fn):
    return os.path.join(ROI_ROOT, cls, fn)

def load_gray(path):
    return resize(imread(path, as_gray=True), CANON, anti_aliasing=True)

def paths_of(sdir, part, c):
    fs = sorted(glob.glob(os.path.join(sdir, part, c, "*")))
    return [roi_path(sdir, c, os.path.basename(f)) for f in fs]

def split_classes(sdir):
    root = os.path.join(sdir, "train")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

def znorm(s):
    s = np.asarray(s)
    return (s - s.mean()) / (s.std() + 1e-8)

# ---------- 4. train内部标定 (KL + w) ----------
def _train_cache(sdir, classes):
    """每类: 灰度图列表 + blpoc变换 + wccd特征 + 全量模板。"""
    cache = {}
    for c in classes:
        grays = [load_gray(p) for p in paths_of(sdir, "train", c)]
        tfs = [blpoc_transform(g) for g in grays]
        tpl = np.mean(grays, axis=0)
        cache[c] = {
            "grays": grays,
            "tfs": tfs,
            "g_tf": blpoc_transform(tpl),
            "g_wccd": wccd_feature(tpl),
        }
    return cache

def calibrate(sdir):
    classes = split_classes(sdir)
    cache = _train_cache(sdir, classes)
    shape = cache[classes[0]]["grays"][0].shape
    G_full = np.stack([cache[c]["g_tf"] for c in classes])
    gw_full = np.stack([cache[c]["g_wccd"] for c in classes])

    # --- (a) 选KL: BLPOC-only 留一 rank-1 ---
    best_kl, best_kl_r = None, -1
    for kl in KL_CANDIDATES:
        mask, B, MN = band_params(shape, kl)
        top1, n = 0, 0
        for ci, c in enumerate(classes):
            d = cache[c]
            for k in range(len(d["grays"])):
                others = [d["tfs"][j] for j in range(len(d["tfs"])) if j != k]
                gb_k = blpoc_transform(np.mean(
                    [d["grays"][j] for j in range(len(d["grays"])) if j != k], axis=0))
                G = G_full.copy(); G[ci] = gb_k
                sb = blpoc_scores_batch(d["tfs"][k], G, mask, B, MN)
                if int(np.argmax(sb)) == ci:
                    top1 += 1
                n += 1
        r = top1 / n
        print(f"  KL={kl:.2f} train_rank1={r*100:.2f}%")
        if r > best_kl_r:
            best_kl_r, best_kl = r, kl
    print(f"selected KL={best_kl:.2f}")

    # --- (b) 选w: 固定KL, 留一融合 rank-1 ---
    mask, B, MN = band_params(shape, best_kl)
    pre = []
    for ci, c in enumerate(classes):
        d = cache[c]
        for k in range(len(d["grays"])):
            others = [d["grays"][j] for j in range(len(d["grays"])) if j != k]
            gb_k = blpoc_transform(np.mean(others, axis=0))
            gw_k = wccd_feature(np.mean(others, axis=0))
            G = G_full.copy(); G[ci] = gb_k
            gw = gw_full.copy(); gw[ci] = gw_k
            sb = blpoc_scores_batch(d["tfs"][k], G, mask, B, MN)
            sw = wccd_sim_batch(wccd_feature(d["grays"][k]), gw)
            pre.append((ci, sb, sw))
    best_w, best_w_r = 0.5, -1
    for w in np.arange(0, 1.01, 0.1):
        top1 = sum(1 for ci, sb, sw in pre
                   if int(np.argmax(w * znorm(sb) + (1 - w) * znorm(sw))) == ci)
        r = top1 / len(pre)
        print(f"  w={w:.1f} train_rank1={r*100:.2f}%")
        if r > best_w_r:
            best_w_r, best_w = r, w
    print(f"selected w={best_w:.1f} (KL={best_kl:.2f})")
    return best_kl, best_w

# ---------- 5. test评测 ----------
def run_split(sdir, kl, w):
    classes = split_classes(sdir)
    assert len(classes) == 360, sdir
    cache = _train_cache(sdir, classes)
    shape = cache[classes[0]]["grays"][0].shape
    mask, B, MN = band_params(shape, kl)
    G = np.stack([cache[c]["g_tf"] for c in classes])
    gw = np.stack([cache[c]["g_wccd"] for c in classes])

    probes = [(c, p) for c in classes for p in paths_of(sdir, "test", c)]
    top1, sp, sn, fails = 0, [], [], []
    for c, p in tqdm(probes, desc=f"probe {os.path.basename(sdir)}", leave=False):
        img = load_gray(p)
        sb = blpoc_scores_batch(blpoc_transform(img), G, mask, B, MN)
        sw = wccd_sim_batch(wccd_feature(img), gw)
        fused = w * znorm(sb) + (1 - w) * znorm(sw)
        pred = classes[int(np.argmax(fused))]
        if pred == c:
            top1 += 1
        else:
            fails.append(f"{c}\t{pred}\t{p}")
        for g, s in zip(classes, fused):
            (sp if g == c else sn).append(s)
    rank1 = top1 / len(probes)
    pos, neg = np.array(sp), np.array(sn)
    thrs = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 2000)
    far = np.array([(neg >= t).mean() for t in thrs])
    frr = np.array([(pos < t).mean() for t in thrs])
    i = int(np.argmin(np.abs(far - frr)))
    return rank1, (far[i] + frr[i]) / 2, fails

def main():
    print("calibrating KL + w on train (test untouched) ...")
    kl, w = calibrate(SPLITS[0])
    r1s, eers = [], []
    for sdir in SPLITS:
        r1, eer, fails = run_split(sdir, kl, w)
        r1s.append(r1 * 100); eers.append(eer * 100)
        print(f"{os.path.basename(sdir)}: Rank-1={r1*100:.2f}%  EER={eer*100:.2f}%  fails={len(fails)}")
        with open(f"asaari_utfvp_failures_{os.path.basename(sdir)}.txt", "w") as f:
            f.write("\n".join(fails))
    print(f"Asaari UTFVP (KL={kl:.2f}, w={w:.1f}, train-calibrated): "
          f"Rank-1={np.mean(r1s):.2f}±{np.std(r1s):.2f}%  "
          f"EER={np.mean(eers):.2f}±{np.std(eers):.2f}%")

if __name__ == "__main__":
    main()
