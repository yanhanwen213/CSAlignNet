"""
Finger-vein ROI from raw NIR images.

Two modes:
  bbox (recommended for CSAlign):
    detect finger → axis-aligned rectangle crop → resize
    (NO orientation correction; AlignmentHead handles rot/shift)

  orient (legacy / classical full ROI):
    detect → minAreaRect rotate upright → crop → tip-down → resize
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class ROIResult:
    roi: np.ndarray  # uint8 HxW
    angle_deg: float
    center: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    flipped_180: bool
    mode: str


def parse_out_size(values: list[int] | tuple[int, ...] | int) -> tuple[int, int]:
    """Accept 112 | (112,112) | [100,300] → (width, height)."""
    if isinstance(values, int):
        return int(values), int(values)
    vals = list(values)
    if len(vals) == 1:
        return int(vals[0]), int(vals[0])
    if len(vals) == 2:
        return int(vals[0]), int(vals[1])
    raise ValueError("out_size must be one int (square) or two ints (W H)")


class FingerVeinROIExtractor:
    """
    Extract finger ROI from raw grayscale images.

    For pipeline:
      raw → bbox ROI (rectangle) → AlignmentHead → Warper → LRFVNet
    use mode='bbox'.

    out_size=(W,H) is fully configurable (not fixed to 112×112).
    Set resize=False to keep the native crop resolution (ignore out_size).
    """

    def __init__(
        self,
        out_size: tuple[int, int] | int = (112, 112),  # (width, height); configurable
        mode: str = "bbox",  # 'bbox' | 'orient'
        resize: bool = True,
        morph_ksize: int = 7,
        pad_ratio: float = 0.06,
        pad_px: int = 8,
        min_area: float = 800.0,
    ) -> None:
        if mode not in ("bbox", "orient"):
            raise ValueError("mode must be 'bbox' or 'orient'")
        self.mode = mode
        self.resize = bool(resize)
        self.out_w, self.out_h = parse_out_size(out_size)
        self.morph_ksize = int(morph_ksize)
        self.pad_ratio = float(pad_ratio)
        self.pad_px = int(pad_px)
        self.min_area = float(min_area)

    def _maybe_resize(self, crop: np.ndarray) -> np.ndarray:
        if not self.resize:
            return crop
        return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def _finger_mask(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = 255 - bw

        def brightness(mask: np.ndarray) -> float:
            m = mask > 0
            if m.sum() < 100:
                return -1.0
            return float(gray[m].mean())

        bw = inv if brightness(inv) > brightness(bw) else bw
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        return bw

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    def _pad_bbox(
        self, x0: int, y0: int, x1: int, y1: int, W: int, H: int
    ) -> tuple[int, int, int, int]:
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        pad = max(self.pad_px, int(self.pad_ratio * max(bw, bh)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W - 1, x1 + pad)
        y1 = min(H - 1, y1 + pad)
        return x0, y0, x1, y1

    def extract_bbox(self, gray: np.ndarray) -> ROIResult | None:
        """Edge/mask → axis-aligned rectangle only (keep pose for AlignmentHead)."""
        g = gray.astype(np.uint8)
        mask = self._finger_mask(g)
        cnt = self._largest_contour(mask)
        if cnt is None or cv2.contourArea(cnt) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(cnt)
        H, W = g.shape
        x0, y0, x1, y1 = self._pad_bbox(x, y, x + w - 1, y + h - 1, W, H)
        crop = g[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return None
        roi = self._maybe_resize(crop)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return ROIResult(
            roi=roi,
            angle_deg=0.0,
            center=(cx, cy),
            bbox_xyxy=(x0, y0, x1, y1),
            flipped_180=False,
            mode="bbox",
        )

    @staticmethod
    def _tip_should_flip(crop: np.ndarray) -> bool:
        h, w = crop.shape
        if h < 12:
            return False
        t = crop[: h // 3]
        b = crop[2 * h // 3 :]
        thr = max(1, int(0.35 * crop.max()))

        def width(block: np.ndarray) -> float:
            m = block > thr
            if not m.any():
                return float(w)
            widths = []
            for row in m:
                xs = np.where(row)[0]
                if len(xs):
                    widths.append(xs[-1] - xs[0] + 1)
            return float(np.median(widths)) if widths else float(w)

        return width(t) + 1e-3 < width(b) * 0.92

    def extract_orient(self, gray: np.ndarray) -> ROIResult | None:
        """Classical oriented ROI (rotate upright inside extractor)."""
        g = gray.astype(np.uint8)
        mask = self._finger_mask(g)
        cnt = self._largest_contour(mask)
        if cnt is None or cv2.contourArea(cnt) < self.min_area:
            return None

        (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
        if rw > rh:
            rw, rh = rh, rw
            ang += 90.0

        H, W = g.shape
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask_bin = np.zeros_like(g)
        cv2.drawContours(mask_bin, [cnt], -1, 255, -1)
        mask_r = cv2.warpAffine(
            mask_bin, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0
        )
        ys, xs = np.where(mask_r > 0)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        x0, y0, x1, y1 = self._pad_bbox(x0, y0, x1, y1, W, H)
        crop = rot[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return None

        flipped = False
        if self._tip_should_flip(crop):
            crop = cv2.rotate(crop, cv2.ROTATE_180)
            flipped = True

        roi = self._maybe_resize(crop)
        return ROIResult(
            roi=roi,
            angle_deg=float(ang),
            center=(float(cx), float(cy)),
            bbox_xyxy=(x0, y0, x1, y1),
            flipped_180=flipped,
            mode="orient",
        )

    def extract_numpy(self, gray: np.ndarray) -> ROIResult | None:
        if gray.ndim != 2:
            raise ValueError("gray must be HxW")
        if self.mode == "bbox":
            return self.extract_bbox(gray)
        return self.extract_orient(gray)

    def extract_pil(self, img: Image.Image) -> Image.Image | None:
        res = self.extract_numpy(np.asarray(img.convert("L")))
        return None if res is None else Image.fromarray(res.roi)

    def __call__(self, img: Image.Image | np.ndarray) -> Image.Image | None:
        if isinstance(img, np.ndarray):
            g = img if img.ndim == 2 else img[..., 0]
            res = self.extract_numpy(g)
            return None if res is None else Image.fromarray(res.roi)
        return self.extract_pil(img)

    def draw_bbox_overlay(self, gray: np.ndarray) -> tuple[np.ndarray, ROIResult | None]:
        """Return BGR overlay with green rectangle + ROI result (bbox mode)."""
        g = gray.astype(np.uint8)
        # force bbox geometry for visualization
        saved = self.mode
        self.mode = "bbox"
        res = self.extract_bbox(g)
        self.mode = saved
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if res is not None:
            x0, y0, x1, y1 = res.bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return vis, res
