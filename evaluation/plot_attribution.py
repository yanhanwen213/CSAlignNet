"""Generate publication figures from the attribution run."""
import json, math, os, sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(REPO / "csalignnet"))
sys.path.insert(0, str(Path(os.environ.get("LRFVNET_PATH", str(REPO.parent / "LRFVNet")))))
sys.path.insert(0, str(REPO / "evaluation"))
import error_attribution as E
from pidanet.cs_dataset import DualSessionIndex, SessionImageDataset, build_default_transform

OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(exist_ok=True)
ATTRIB = json.load(open(REPO / "evaluation" / "attribution_3seeds.json", encoding="utf-8"))

device = "cuda" if torch.cuda.is_available() else "cpu"
E.set_seed(44)
model, cfg = E.load_model(44, device, os.environ.get(
    "CSALIGN_CKPT_DIR", str(REPO / "checkpoints" / "cs_align")))
idx = DualSessionIndex(cfg["session1_root"], cfg["session2_root"])
gs = SessionImageDataset(cfg["session1_root"], class_names=idx.class_names,
                         class_to_idx=idx.class_to_idx, image_size=112)
ps = SessionImageDataset(cfg["session2_root"], class_names=idx.class_names,
                         class_to_idx=idx.class_to_idx, image_size=112)
tf = build_default_transform(112)
gl = DataLoader(gs, batch_size=64); pl = DataLoader(ps, batch_size=64)
g_emb, g_y = E.extract(model, gl, device)
p_emb, p_y = E.extract(model, pl, device)
templ, cls = E.norm_template(g_emb, g_y)
pe = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-12)
S = pe @ templ.T
pred = cls[S.argmax(1)]

# gallery representative image per class (first sample)
gal_img = {}
for path, y in gs.samples:
    gal_img.setdefault(int(y), str(path))
probe_paths = [str(s[0]) for s in ps.samples]

def gray(p):
    im = Image.open(p).convert("L").resize((112, 112))
    return np.asarray(im, dtype=np.float32) / 255.0

# pick 3 DISTINCT successes (high margin) and 3 DISTINCT pose-recoverable failures
srt_pos = S[np.arange(len(p_y)), [list(cls).index(int(y)) for y in p_y]]
ok_idx = np.where(pred == p_y)[0]
suc, seen = [], set()
for i in ok_idx[np.argsort(-srt_pos[ok_idx])]:
    key = (int(p_y[i]), int(pred[i]))
    if key in seen:
        continue
    seen.add(key); suc.append(i)
    if len(suc) == 3:
        break
fail_idx = np.where(pred != p_y)[0]
seed44 = next(r for r in ATTRIB["per_seed"] if r["seed"] == 44)
det = {(d["probe"], d["true"]): d for d in seed44["detail"]}
pose_fail, seen_f = [], set()
for i in fail_idx:
    key = (Path(probe_paths[i]).name, int(p_y[i]))
    if key in det and det[key]["recovered"]:
        pair = (int(p_y[i]), int(pred[i]))
        if pair in seen_f:
            continue
        seen_f.add(pair); pose_fail.append(i)
        if len(pose_fail) == 3:
            break

def panel(ax, pth, gth, title, color):
    ax.imshow(np.hstack([gray(pth), np.ones((112, 4)), gray(gth)]), cmap="gray")
    ax.set_title(title, color=color, fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(1.2)
    ax.text(0.02, 0.04, "Probe S2", transform=ax.transAxes, color="w",
            fontsize=6, weight="bold")
    ax.text(0.62, 0.04, "Gallery S1", transform=ax.transAxes, color="w",
            fontsize=6, weight="bold")

fig, axes = plt.subplots(2, 3, figsize=(9, 3.6),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.08,
                                          "top": 0.92, "bottom": 0.02,
                                          "left": 0.02, "right": 0.98})
for j, i in enumerate(suc):
    yt = int(p_y[i]); sc = srt_pos[i]
    m = np.sort(S[i])[-1] - np.sort(S[i])[-2]
    panel(axes[0, j], probe_paths[i], gal_img[yt],
          f"$\\checkmark$ {cls[int(pred[i])]} | score {sc:.2f} / margin {m:.2f}",
          "#1a7f37")
for j, i in enumerate(pose_fail):
    yt = int(p_y[i]); yp = int(pred[i])
    panel(axes[1, j], probe_paths[i], gal_img[yt],
          f"$\\times$ true {cls[yt]} / pred {cls[yp]} (pose-recoverable)",
          "#c0392b")
fig.savefig(OUT_DIR / "fig_attrib_matches.png", dpi=200, bbox_inches="tight",
            pad_inches=0.05)
print("saved fig_attrib_matches.png")

# attribution bar (three-seed aggregate)
s = ATTRIB["aggregate"]
cats = ["Out-of-envelope\npose", "Photometric\n/ style drift", "Intrinsically\nambiguous"]
vals = [s["pct_pose"][0], s["pct_photometric"][0], s["pct_intrinsic"][0]]
fig2, ax = plt.subplots(figsize=(5, 3.6))
bars = ax.bar(cats, vals, color=["#c0392b", "#2980b9", "#7f8c8d"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
            ha="center", fontsize=10, weight="bold")
ax.set_ylabel("Share of residual errors (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Error attribution (FV-USM, 3-seed mean)", fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "fig_attrib_bars.png", dpi=200)
print("saved fig_attrib_bars.png")
