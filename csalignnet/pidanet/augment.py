"""Photometric / Fourier pseudo-session augmentation (PIL, grayscale).

FDA: Yang & Soatto, CVPR 2020 — swap low-frequency amplitude between images.
MixStyle/DSU/pAdaIN are feature-level and live in style_layers.py.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def pil_fda(
    source: Image.Image,
    donor: Image.Image,
    beta: float = 0.05,
) -> Image.Image:
    """Fourier Domain Adaptation: swap low-frequency amplitude.

    Keeps phase (structure/veins) of source, replaces low-frequency
    amplitude with that of donor.  beta ∈ (0,0.5) controls low-frequency
    window size.  Ref: Yang & Soatto, FDA, CVPR 2020.
    No Session-2 data is used; donor is another Session-1 image.
    """
    import cv2  # noqa: F401  # ensure available; not used directly

    src = np.asarray(source.convert("L"), dtype=np.float32)
    ref = np.asarray(donor.convert("L"), dtype=np.float32)
    # donor may differ in size (original ROI); resize to source size
    if ref.shape != src.shape:
        ref = np.asarray(donor.convert("L").resize(source.size, Image.BILINEAR), dtype=np.float32)
    h, w = src.shape
    b = float(np.clip(beta, 0.005, 0.45))
    bh, bw = max(1, int(h * b)), max(1, int(w * b))
    cy, cx = h // 2, w // 2

    fft_src = np.fft.fftshift(np.fft.fft2(src))
    fft_ref = np.fft.fftshift(np.fft.fft2(ref))
    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_ref = np.abs(fft_ref)

    amp_src[cy - bh : cy + bh, cx - bw : cx + bw] = amp_ref[cy - bh : cy + bh, cx - bw : cx + bw]

    fft_new = amp_src * np.exp(1j * pha_src)
    out = np.fft.ifft2(np.fft.ifftshift(fft_new)).real
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def pil_pseudo_session(
    img: Image.Image,
    brightness: tuple[float, float] = (0.55, 1.45),
    contrast: tuple[float, float] = (0.55, 1.45),
    gamma: tuple[float, float] = (0.65, 1.45),
    blur_prob: float = 0.35,
    blur_radius: tuple[float, float] = (0.4, 1.2),
    noise_std: float = 6.0,
    noise_prob: float = 0.5,
) -> Image.Image:
    """
    Simulate cross-session appearance shift without using Session-2 images.

    Applied to the contrastive partner view (x_b) under s1_only.
    Geometry (rot/shift) is handled separately by pose augmentation.
    """
    out = img.convert("L")

    out = ImageEnhance.Brightness(out).enhance(random.uniform(*brightness))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(*contrast))

    g = random.uniform(*gamma)
    # Pillow point table: 256 entries for 8-bit
    lut = [min(255, int(((i / 255.0) ** g) * 255.0 + 0.5)) for i in range(256)]
    out = out.point(lut)

    if random.random() < blur_prob:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur_radius)))

    if noise_std > 0 and random.random() < noise_prob:
        arr = np.asarray(out, dtype=np.float32)
        arr = arr + np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr, mode="L")

    return out
