"""
Asaari et al. baseline: BLPOC + WCCD score-level fusion.
Asaari et al., "Fusion of band-limited phase-only correlation and width
centroid contour distance for finger based biometrics", ESWA 2014.

严格协议: Gallery=S1 (类均值模板), Probe=S2; 融合权重仅在S1-val上标定, 不碰S2.
"""
import glob, os, numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.transform import resize

# FV-USM root: set FVUSM_ROOT or edit here.
DB = os.environ.get("FVUSM_ROOT", r"<path-to-FV-USM>")
CANON_H = 112
KL = 0.30            # BLPOC带限比例 (Asaari默认低通带)
WCCD_SCALE = 0.5     # WCCD距离->相似度尺度

# ---------- 1. BLPOC ----------
def band_mask(shape, kl=KL):
    M, N = shape
    m, n = max(1, int(M * kl / 2)), max(1, int(N * kl / 2))
    mask = np.zeros(shape)
    mask[:m, :n] = mask[-m:, :n] = mask[:m, -n:] = mask[-m:, -n:] = 1.0
    return mask

def blpoc_transform(img):
    """返回相位谱 F/|F|，供批量相关。"""
    F = np.fft.fft2(img - img.mean())
    return F / (np.abs(F) + 1e-8)

def blpoc_score(prep_q, prep_g, mask, B, MN):
    """相位相关峰值，归一化到~[0,1]。"""
    R = prep_q * np.conj(prep_g)
    R = np.fft.ifft2(R * mask)
    return float(np.abs(R).max() * MN / B)

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
    h = gray.shape[0]
    center = np.full(h, np.nan); width = np.full(h, np.nan)
    for y in np.where(m.any(axis=1))[0]:
        xs = np.where(m[y])[0]
        center[y] = (xs[0] + xs[-1]) / 2
        width[y] = xs[-1] - xs[0]
    cx = np.nanmean(center); mw = np.nanmean(width) + 1e-8
    center = np.nan_to_num(center - cx) / mw
    width = np.nan_to_num(width) / mw
    return np.concatenate([center, width])

def wccd_sim(f1, f2):
    d = np.linalg.norm(f1 - f2) / len(f1)
    return float(1.0 / (1.0 + d / WCCD_SCALE))

# ---------- 3. 数据 ----------
def load_session(sess_dir):
    db = {}
    for cls in sorted(os.listdir(sess_dir)):
        cdir = os.path.join(sess_dir, cls)
        if not os.path.isdir(cdir):
            continue
        imgs = []
        for fn in sorted(glob.glob(os.path.join(cdir, "*.jpg"))):
            img = imread(fn, as_gray=True)
            h, w = img.shape
            img = resize(img, (CANON_H, int(CANON_H * w / h)), anti_aliasing=True)
            imgs.append(img)
        if imgs:
            db[cls] = imgs
    return db

def build_gallery(imgs_list):
    tpl_img = np.mean(imgs_list, axis=0)              # 均值灰度模板 (BLPOC)
    return blpoc_transform(tpl_img), wccd_feature(np.mean(imgs_list, axis=0))

# ---------- 4. 分数 z-normalize + 融合 ----------
def znorm(scores):
    s = np.asarray(scores)
    return (s - s.mean()) / (s.std() + 1e-8)

def eval_with_weight(classes, g_blpoc, g_wccd, g_feats, probes, mask, B, MN, w,
                     precomputed=None):
    if precomputed is not None:
        pre = precomputed
    else:
        pre = []
        for c, pimg in probes:
            pq, pw = blpoc_transform(pimg), wccd_feature(pimg)
            sb = np.array([blpoc_score(pq, g_blpoc[g], mask, B, MN) for g in classes])
            sw = np.array([wccd_sim(pw, g_wccd[g]) for g in classes])
            pre.append((c, sb, sw))
    top1, n = 0, 0
    for c, sb, sw in pre:
        fused = w * znorm(sb) + (1 - w) * znorm(sw)
        if classes[int(np.argmax(fused))] == c:
            top1 += 1
        n += 1
    return top1 / n

def main():
    s1 = load_session(os.path.join(DB, "1st_session", "extractedvein"))
    s2 = load_session(os.path.join(DB, "2nd_session", "extractedvein"))
    classes = sorted(set(s1) & set(s2))
    assert len(classes) == 492, f"类数={len(classes)}"
    print(f"classes={len(classes)}")

    # --- 权重标定: S1留出2张做val, 其余建库, 扫w ---
    sample = s1[classes[0]][0]
    mask = band_mask(sample.shape)
    B, MN = mask.sum(), sample.size
    g_blpoc, g_wccd = {}, {}
    for c in classes:
        g_blpoc[c], g_wccd[c] = build_gallery(s1[c][:4])
    val = [(c, im) for c in classes for im in s1[c][4:]]
    print("precomputing scores for weight calibration ...")
    pre = []
    for c, pimg in val:
        pq, pw = blpoc_transform(pimg), wccd_feature(pimg)
        sb = np.array([blpoc_score(pq, g_blpoc[g], mask, B, MN) for g in classes])
        sw = np.array([wccd_sim(pw, g_wccd[g]) for g in classes])
        pre.append((c, sb, sw))
    print("calibrating fusion weight on S1-val ...")
    best_w, best_r = 0.5, -1
    for w in np.arange(0, 1.01, 0.1):
        r = eval_with_weight(classes, g_blpoc, g_wccd, None, val,
                             mask, B, MN, w, precomputed=pre)
        print(f"  w={w:.1f} val_rank1={r*100:.2f}%")
        if r > best_r:
            best_r, best_w = r, w
    print(f"best_w={best_w:.1f} (val_rank1={best_r*100:.2f}%)")

    # --- 最终: Gallery=S1全6张, Probe=S2 ---
    for c in classes:
        g_blpoc[c], g_wccd[c] = build_gallery(s1[c])

    top1, scores_pos, scores_neg, fails, n = 0, [], [], [], 0
    for c in tqdm(classes, desc="probe"):
        for pimg in s2[c]:
            pq = blpoc_transform(pimg)
            pw = wccd_feature(pimg)
            sb = np.array([blpoc_score(pq, g_blpoc[g], mask, B, MN) for g in classes])
            sw = np.array([wccd_sim(pw, g_wccd[g]) for g in classes])
            fused = best_w * znorm(sb) + (1 - best_w) * znorm(sw)
            pred = classes[int(np.argmax(fused))]
            n += 1
            if pred == c:
                top1 += 1
            else:
                fails.append(f"{c}\t{pred}")
            for g, s in zip(classes, fused):
                (scores_pos if g == c else scores_neg).append(s)
    rank1 = top1 / n
    pos = np.array(scores_pos); neg = np.array(scores_neg)
    thrs = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 2000)
    far = np.array([(neg >= t).mean() for t in thrs])
    frr = np.array([(pos < t).mean() for t in thrs])
    i = int(np.argmin(np.abs(far - frr)))
    eer = (far[i] + frr[i]) / 2
    print(f"Asaari BLPOC+WCCD STRICT: Rank-1={rank1*100:.2f}%  "
          f"EER={eer*100:.2f}%  (n={n}, w={best_w:.1f})")
    with open("asaari_failures.txt", "w") as f:
        f.write("\n".join(fails))

if __name__ == "__main__":
    main()
