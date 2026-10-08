"""
Visualize rigid rot/shift on official ROI at native resolution
(AFTER geometry, BEFORE resize to 112).

Example:
  python vis_roi_pose_native.py ^
    --image <session1>/extractedvein/<class>/01.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pil_rotate_translate(img: Image.Image, angle: float, dx: float, dy: float) -> Image.Image:
    out = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return out.transform(
        out.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=0,
    )


def main():
    p = argparse.ArgumentParser(
        description="Visualize ROI pose AFTER geometry, BEFORE network resize"
    )
    p.add_argument("--image", type=str, default="")
    p.add_argument(
        "--session1_root",
        type=str,
        default="",
        help="Session-1 extractedvein root (or set FVUSM_S1)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "log" / "vis_pose_native"),
    )
    p.add_argument(
        "--net_size",
        type=int,
        default=112,
        help="Only used for optional side panel; main panels stay native",
    )
    args = p.parse_args()

    if args.image:
        img_path = Path(args.image)
    else:
        root = Path(args.session1_root)
        cls = sorted(d for d in root.iterdir() if d.is_dir())[0]
        imgs = sorted(
            q
            for q in cls.iterdir()
            if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        img_path = imgs[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Native ROI (no resize yet) — FV-USM extractedvein is typically 100×300
    native = Image.open(img_path).convert("L")
    nw, nh = native.size

    panels: list[tuple[str, Image.Image]] = [
        (f"Original native ROI\n{nw}×{nh}", native),
        ("AFTER geo: rot +15°\n(before resize)", pil_rotate_translate(native, 15, 0, 0)),
        ("AFTER geo: rot -15°\n(before resize)", pil_rotate_translate(native, -15, 0, 0)),
        ("AFTER geo: rot +25°\n(before resize)", pil_rotate_translate(native, 25, 0, 0)),
        ("AFTER geo: shift +15 px\n(before resize)", pil_rotate_translate(native, 0, 15, 0)),
        ("AFTER geo: shift -15 px\n(before resize)", pil_rotate_translate(native, 0, -15, 0)),
        ("AFTER geo: dy +15\n(before resize)", pil_rotate_translate(native, 0, 0, 15)),
        (
            "AFTER geo: rot15+shift(10,8)\n(before resize)",
            pil_rotate_translate(native, 15, 10, 8),
        ),
        (
            "AFTER geo: rot-20+shift(-12,10)\n(before resize)",
            pil_rotate_translate(native, -20, -12, 10),
        ),
    ]

    # Save individual native-resolution panels
    for i, (title, im) in enumerate(panels):
        assert im.size == (nw, nh), "geometry must keep native size (expand=False)"
        safe = (
            title.split("\n")[0]
            .replace(" ", "_")
            .replace("°", "deg")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace("+", "p")
        )
        im.save(out_dir / f"{i:02d}_{safe}_native.png")

    # Grid: all at native aspect (tall ROI)
    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig_w = 2.2 * ncols
    fig_h = 3.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=160)
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, im) in zip(axes, panels):
        ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(f"{im.size[0]}×{im.size[1]}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"Rigid pose on native ROI (geometry → then resize)\n{img_path}",
        fontsize=10,
    )
    fig.tight_layout()
    grid_path = out_dir / "roi_pose_native_before_resize.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)

    # Extra strip: one example native geo vs after resize (to contrast)
    geo = pil_rotate_translate(native, 15, 10, 8)
    resized = geo.resize((args.net_size, args.net_size), Image.BILINEAR)
    fig2, ax2 = plt.subplots(1, 2, figsize=(5.5, 3.2), dpi=160)
    ax2[0].imshow(np.asarray(geo), cmap="gray", vmin=0, vmax=255)
    ax2[0].set_title(f"AFTER geo, BEFORE resize\n{geo.size[0]}×{geo.size[1]}", fontsize=9)
    ax2[0].axis("off")
    ax2[1].imshow(np.asarray(resized), cmap="gray", vmin=0, vmax=255)
    ax2[1].set_title(f"AFTER resize (network input)\n{args.net_size}×{args.net_size}", fontsize=9)
    ax2[1].axis("off")
    fig2.suptitle("Same pose: native geo vs resized", fontsize=10)
    fig2.tight_layout()
    cmp_path = out_dir / "native_geo_vs_resize.png"
    fig2.savefig(cmp_path, bbox_inches="tight")
    plt.close(fig2)

    print(f"source: {img_path}")
    print(f"native size: {nw}x{nh}")
    print(f"grid: {grid_path}")
    print(f"compare: {cmp_path}")
    print(f"dir: {out_dir}")


if __name__ == "__main__":
    main()
