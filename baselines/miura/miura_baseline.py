import glob, os, numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects
from scipy.ndimage import convolve1d

# FV-USM root: set FVUSM_ROOT or edit here.
DB = os.environ.get("FVUSM_ROOT", r"<path-to-FV-USM>")
THR_K = float(os.environ.get("MIURA_K", "1.5"))  # 曲率阈值系数，多轮测试时切换

# ---------- 1. Miura最大曲率静脉提取 (Miura et al. 2007) ----------
def max_curvature_vein(gray):
    """输入灰度ROI(黑暗=静脉)，输出二值静脉图。"""
    g = gray.astype(np.float64)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    score = np.zeros_like(g)
    for s in [1.0, 2.0, 3.0]:
        f = gaussian(g, s)
        for axis in (0, 1):
            d1 = convolve1d(f, [-0.5, 0, 0.5], axis=axis)
            d2 = convolve1d(f, [1, -2, 1], axis=axis)
            k = np.abs(d2) / (1 + d1 ** 2) ** 1.5
            score = np.maximum(score, k * (d2 > 0))
        fr = np.rot90(f)
        d2r = convolve1d(fr, [1, -2, 1], axis=0)
        score = np.maximum(score, np.rot90(np.abs(d2r) * (d2r > 0), 3))
    thr = score.mean() + THR_K * score.std()
    bw = remove_small_objects(score > thr, min_size=30)
    return bw.astype(np.float64)

def load_session(sess_dir):
    """{class: [img,...]}，保持高宽比，高度缩放到112。"""
    db = {}
    for cls in sorted(os.listdir(sess_dir)):
        cdir = os.path.join(sess_dir, cls)
        if not os.path.isdir(cdir):
            continue
        imgs = []
        for fn in sorted(glob.glob(os.path.join(cdir, "*.jpg"))):
            img = imread(fn, as_gray=True)
            h, w = img.shape
            img = resize(img, (112, int(112 * w / h)), anti_aliasing=True)
            imgs.append((fn, img))
        if imgs:
            db[cls] = imgs
    return db

SHIFTS = [(dy, dx) for dy in range(-6, 7, 2) for dx in range(-6, 7, 2)]  # 49个位移

def shift_zero(q, dy, dx):
    """零填充平移(不卷绕)，模拟手指摆放位移。"""
    h, w = q.shape
    out = np.zeros_like(q)
    out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] = \
        q[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out

# ---------- 2. 严格协议评测: Gallery=S1类均值模板, Probe=S2 ----------
def evaluate():
    s1 = load_session(os.path.join(DB, "1st_session", "extractedvein"))
    s2 = load_session(os.path.join(DB, "2nd_session", "extractedvein"))
    classes = sorted(set(s1) & set(s2))
    print(f"classes={len(classes)}, "
          f"S1/class={np.mean([len(s1[c]) for c in classes]):.1f}, "
          f"S2/class={np.mean([len(s2[c]) for c in classes]):.1f}")
    assert len(classes) == 492, "类数不是492，数据不对"

    gallery = {c: (np.mean([max_curvature_vein(im) for _, im in
                            tqdm(s1[c], desc="gallery", leave=False)], axis=0) > 0.5
                   ).astype(float) for c in tqdm(classes, desc="templates")}

    Gmat = np.stack([(gallery[c] - gallery[c].mean()).ravel() for c in classes])
    Gmat /= (np.linalg.norm(Gmat, axis=1, keepdims=True) + 1e-8)
    D = Gmat.shape[1]

    top1_ok, all_scores, gen_idx, fails, n = 0, [], [], [], 0
    for c in tqdm(classes, desc="probe"):
        for ppath, pim in s2[c]:
            q0 = max_curvature_vein(pim)
            best = np.full(len(classes), -np.inf)
            for dy, dx in SHIFTS:
                qs = shift_zero(q0, dy, dx).ravel()
                qs = qs - qs.mean()
                qs /= (np.linalg.norm(qs) + 1e-8)
                np.maximum(best, Gmat @ qs, out=best)
            pred = classes[int(np.argmax(best))]
            n += 1
            if pred == c:
                top1_ok += 1
            else:
                fails.append(f"{c}\t{pred}\t{ppath}")
            all_scores.append(best)
            gen_idx.append(classes.index(c))

    scores = np.array(all_scores)  # (2952, 492)
    rank1 = top1_ok / n
    pos = scores[np.arange(n), gen_idx]
    mask = np.ones_like(scores, dtype=bool)
    mask[np.arange(n), gen_idx] = False
    neg = scores[mask]
    thrs = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 2000)
    far = np.array([(neg >= t).mean() for t in thrs])
    frr = np.array([(pos < t).mean() for t in thrs])
    i = int(np.argmin(np.abs(far - frr)))
    eer = (far[i] + frr[i]) / 2
    print(f"Rank-1: {rank1*100:.2f}%   EER: {eer*100:.2f}%   (n={n})")
    with open("miura_failures.txt", "w") as f:
        f.write("\n".join(fails))
    print(f"失败样本 {len(fails)} 个，已存 miura_failures.txt")

if __name__ == "__main__":
    evaluate()
