"""
Cross-session datasets for CSAlignNet.

Preferred (true sessions, FV-USM):
  session1_root = .../1st_session/raw_data
  session2_root = .../2nd_session/raw_data
  class folder names must match (e.g. 001_1).

Single-root HK-style filenames:
  {id}_{cap}_f{finger}_{session}_orig.jpg  → session digit before _orig/_00
"""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

_LRFV = Path(__file__).resolve().parents[2] / "LRFVNet"
if _LRFV.is_dir() and str(_LRFV) not in sys.path:
    sys.path.insert(0, str(_LRFV))

from lrfvnet.dataset import FingerVeinFolder, build_default_transform  # noqa: E402
from pidanet.augment import pil_fda  # noqa: E402  (FDA CVPR20)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def _pil_rotate_translate(img: Image.Image, angle: float, dx: float, dy: float) -> Image.Image:
    out = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return out.transform(
        out.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=0,
    )


def _pil_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Contrast-Limited Adaptive Histogram Equalization (Zuiderveld, Graphics Gems IV).

    Applied identically in train and eval to enhance local vein contrast and
    reduce illumination mismatch across sessions. Deterministic (no randomness).
    """
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    out = clahe.apply(arr)
    return Image.fromarray(out, mode="L")


def _pil_pseudo_session(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Photometric 'pseudo-session' augmentation (no extra rigid warp).

    Simulates acquisition / illumination shift between sessions so InfoNCE
    under s1_only sees a harder second view than another clean S1 capture.
    strength=1.0 is the default operating point; 0 disables (caller should skip).
    """
    s = float(max(0.0, strength))
    if s <= 0:
        return img
    out = img
    # Brightness / contrast (wider with larger strength)
    b_lo, b_hi = 1.0 - 0.35 * s, 1.0 + 0.35 * s
    c_lo, c_hi = 1.0 - 0.40 * s, 1.0 + 0.40 * s
    out = ImageEnhance.Brightness(out).enhance(random.uniform(b_lo, b_hi))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(c_lo, c_hi))
    # Gamma
    gamma = random.uniform(max(0.45, 1.0 - 0.45 * s), min(1.8, 1.0 + 0.55 * s))
    lut = [min(255, int(((i / 255.0) ** gamma) * 255.0 + 0.5)) for i in range(256)]
    out = out.point(lut)
    # Occasional mild blur
    if random.random() < 0.35 * min(s, 1.5):
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9 * s)))
    # Light sensor noise
    if random.random() < 0.5 * min(s, 1.5):
        arr = np.asarray(out, dtype=np.float32)
        sigma = random.uniform(1.5, 6.0) * s
        arr = np.clip(arr + np.random.randn(*arr.shape) * sigma, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8), mode="L")
    return out


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _prefer_orig(paths: list[Path]) -> list[Path]:
    orig = [p for p in paths if "orig" in p.stem.lower() or re.fullmatch(r"\d+", p.stem)]
    return orig if orig else paths


_HK_SESS_RE = re.compile(
    r"^(?P<prefix>.+)_f(?P<finger>\d+)_(?P<session>\d+)(?:_|$)", re.IGNORECASE
)


def parse_hk_session(name: str) -> int | None:
    """Return 1/2 if filename encodes session, else None."""
    stem = Path(name).stem
    # strip trailing aug tags like _00_r... 
    m = _HK_SESS_RE.match(stem)
    if m:
        return int(m.group("session"))
    # fallback: ..._1_orig / ..._2_orig
    m2 = re.search(r"_([12])(?:_orig)?$", stem)
    if m2:
        return int(m2.group(1))
    return None


def _normalize_class_key(name: str) -> str:
    """Map 'vein001_1' / '001_1' → '001_1' for cross-root matching."""
    n = name.strip()
    if n.lower().startswith("vein"):
        n = n[4:]
    return n


class DualSessionIndex:
    """Map class_name -> (s1_paths, s2_paths, label)."""

    def __init__(
        self,
        session1_root: Path,
        session2_root: Path,
        prefer_orig: bool = True,
    ) -> None:
        self.session1_root = Path(session1_root)
        self.session2_root = Path(session2_root)
        self.prefer_orig = prefer_orig

        def _folder_map(root: Path) -> dict[str, str]:
            """normalized_key -> actual folder name on disk."""
            out: dict[str, str] = {}
            for d in root.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    out[_normalize_class_key(d.name)] = d.name
            return out

        m1 = _folder_map(self.session1_root)
        m2 = _folder_map(self.session2_root)
        common_keys = sorted(set(m1) & set(m2))
        if not common_keys:
            raise RuntimeError(
                f"No common class folders between {session1_root} and {session2_root} "
                f"(after normalizing names like vein001_1 ↔ 001_1). "
                f"S1 eg={list(m1.values())[:3]} S2 eg={list(m2.values())[:3]}"
            )
        # Use normalized keys as canonical class names
        self.class_names = common_keys
        self.class_to_idx = {n: i for i, n in enumerate(common_keys)}
        self.entries: list[tuple[str, list[Path], list[Path], int]] = []
        for key in common_keys:
            folder1 = self.session1_root / m1[key]
            folder2 = self.session2_root / m2[key]
            p1 = _list_images(folder1)
            p2 = _list_images(folder2)
            if prefer_orig:
                p1 = _prefer_orig(p1)
                p2 = _prefer_orig(p2)
            if not p1 or not p2:
                continue
            self.entries.append((key, p1, p2, self.class_to_idx[key]))
        if not self.entries:
            raise RuntimeError("No class has images in both sessions.")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class CrossSessionPoseDataset(Dataset):
    """
    Joint dual-session training (weaker protocol).

    Returns:
      x_s1, x_s2, x_pose, x_pos, x_neg, label

    x_pose = rigid augmentation of x_s1 (or random session view).
    """

    def __init__(
        self,
        session1_root: str | Path,
        session2_root: str | Path,
        transform=None,
        image_size: int = 112,
        rot_range: tuple[float, float] = (-30.0, 30.0),
        shift_range: tuple[float, float] = (-20.0, 20.0),
        class_indices: list[int] | None = None,
        pose_from: str = "s1",
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
    ) -> None:
        self.index = DualSessionIndex(Path(session1_root), Path(session2_root))
        self.transform = transform or build_default_transform(image_size)
        self.image_size = image_size
        self.rot_range = rot_range
        self.shift_range = shift_range
        self.pose_from = pose_from
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = (int(clahe_tile_grid[0]), int(clahe_tile_grid[1]))
        self.entries = self.index.entries
        if class_indices is not None:
            allow = set(class_indices)
            self.entries = [e for e in self.entries if e[3] in allow]
        if not self.entries:
            raise RuntimeError("Empty CrossSessionPoseDataset after class filter.")
        self.label_to_entries: dict[int, list] = defaultdict(list)
        for e in self.entries:
            self.label_to_entries[e[3]].append(e)

    @property
    def num_classes(self) -> int:
        return self.index.num_classes

    def __len__(self) -> int:
        return len(self.entries)

    def _load_native(self, path: Path) -> Image.Image:
        """Load at original ROI resolution (geometry before resize)."""
        img = Image.open(path).convert("L")
        if self.use_clahe:
            img = _pil_clahe(
                img,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid,
            )
        return img

    def _rand_geo(self) -> tuple[float, float, float]:
        return (
            random.uniform(*self.rot_range),
            random.uniform(*self.shift_range),
            random.uniform(*self.shift_range),
        )

    def __getitem__(self, index: int):
        _name, p1, p2, label = self.entries[index]
        path_s1 = random.choice(p1)
        path_s2 = random.choice(p2)
        pil_s1 = self._load_native(path_s1)
        pil_s2 = self._load_native(path_s2)

        # Rigid aug on native resolution, then transform resizes to image_size
        base = pil_s1 if self.pose_from == "s1" else random.choice([pil_s1, pil_s2])
        ang, dx, dy = self._rand_geo()
        pil_pose = _pil_rotate_translate(base, ang, dx, dy)

        # positive: other image of same id (prefer other session)
        pos_pool = [p for p in (p1 + p2) if p != path_s1]
        if not pos_pool:
            pos_pool = p2
        pil_pos = self._load_native(random.choice(pos_pool))

        neg_label = random.choice([c for c in self.label_to_entries if c != label])
        _n, np1, np2, _ = random.choice(self.label_to_entries[neg_label])
        pil_neg = self._load_native(random.choice(np1 + np2))

        return (
            self.transform(pil_s1),
            self.transform(pil_s2),
            self.transform(pil_pose),
            self.transform(pil_pos),
            self.transform(pil_neg),
            label,
        )


class Session1OnlyPoseDataset(Dataset):
    """
    Strict protocol: train on Session-1 images only (S2 never seen in training).

    Returns same 6-tuple layout as CrossSessionPoseDataset:
      x_a, x_b, x_pose, x_pos, x_neg, label
    where x_a is a clean S1 capture and x_b is another S1 capture
    (optionally with photometric pseudo-session augmentation).
    All paths are drawn from Session-1 only.
    """

    def __init__(
        self,
        session1_root: str | Path,
        session2_root: str | Path,
        transform=None,
        image_size: int = 112,
        rot_range: tuple[float, float] = (-30.0, 30.0),
        shift_range: tuple[float, float] = (-20.0, 20.0),
        class_indices: list[int] | None = None,
        prefer_orig: bool = True,
        pseudo_session: bool = True,
        pseudo_session_strength: float = 1.0,
        # FDA: Fourier Domain Adaptation (Yang & Soatto, CVPR 2020)
        fda: bool = False,
        fda_prob: float = 0.5,
        fda_beta: float = 0.05,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
    ) -> None:
        # DualSessionIndex keeps class list aligned with S2 for later eval.
        self.index = DualSessionIndex(
            Path(session1_root), Path(session2_root), prefer_orig=prefer_orig
        )
        self.transform = transform or build_default_transform(image_size)
        self.image_size = image_size
        self.rot_range = rot_range
        self.shift_range = shift_range
        self.pseudo_session = bool(pseudo_session)
        self.pseudo_session_strength = float(pseudo_session_strength)
        self.fda = bool(fda)
        self.fda_prob = float(fda_prob)
        self.fda_beta = float(fda_beta)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = (int(clahe_tile_grid[0]), int(clahe_tile_grid[1]))

        self.entries = self.index.entries
        if class_indices is not None:
            allow = set(class_indices)
            self.entries = [e for e in self.entries if e[3] in allow]
        if not self.entries:
            raise RuntimeError("Empty Session1OnlyPoseDataset after class filter.")

        # One training sample per S1 image (anchor).
        self.samples: list[tuple[list[Path], Path, int]] = []
        self.label_to_s1: dict[int, list[Path]] = defaultdict(list)
        for _name, p1, _p2, label in self.entries:
            self.label_to_s1[label].extend(p1)
            for p in p1:
                self.samples.append((p1, p, label))
        if not self.samples:
            raise RuntimeError("No Session-1 images for Session1OnlyPoseDataset.")

    @property
    def num_classes(self) -> int:
        return self.index.num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def _load_native(self, path: Path) -> Image.Image:
        """Load at original ROI resolution; optional CLAHE before geometry/aug."""
        img = Image.open(path).convert("L")
        if self.use_clahe:
            img = _pil_clahe(
                img,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid,
            )
        return img

    def _rand_geo(self) -> tuple[float, float, float]:
        return (
            random.uniform(*self.rot_range),
            random.uniform(*self.shift_range),
            random.uniform(*self.shift_range),
        )

    def __getitem__(self, index: int):
        p1, path_a, label = self.samples[index]
        pil_a = self._load_native(path_a)

        others = [p for p in p1 if p != path_a]
        if others:
            path_b = random.choice(others)
            pil_b = self._load_native(path_b)
        else:
            # Single image in class: second view = mild re-aug of the same capture
            ang0, dx0, dy0 = self._rand_geo()
            pil_b = _pil_rotate_translate(pil_a, ang0 * 0.3, dx0 * 0.3, dy0 * 0.3)

        # Pseudo-session: photometric shift on x_b only (x_a stays clean).
        # Optional FDA: low-frequency amplitude swap with another S1 image
        # (no S2 data; donor sampled from any S1 image in the training set).
        if self.fda and random.random() < self.fda_prob:
            # sample a donor S1 image uniformly from training pool
            donor_label = random.choice(list(self.label_to_s1.keys()))
            donor_path = random.choice(self.label_to_s1[donor_label])
            donor_img = self._load_native(donor_path)
            # FDA first, then photometric; FDA preserves structure via phase
            try:
                pil_b = pil_fda(pil_b, donor_img, beta=self.fda_beta)
            except Exception:
                pass  # fallback to photometric only
        if self.pseudo_session and self.pseudo_session_strength > 0:
            pil_b = _pil_pseudo_session(pil_b, strength=self.pseudo_session_strength)

        # Rigid aug on native resolution, then transform resizes to image_size
        ang, dx, dy = self._rand_geo()
        pil_pose = _pil_rotate_translate(pil_a, ang, dx, dy)

        pos_pool = others if others else p1
        pil_pos = self._load_native(random.choice(pos_pool))

        neg_label = random.choice([c for c in self.label_to_s1 if c != label])
        pil_neg = self._load_native(random.choice(self.label_to_s1[neg_label]))

        return (
            self.transform(pil_a),
            self.transform(pil_b),
            self.transform(pil_pose),
            self.transform(pil_pos),
            self.transform(pil_neg),
            label,
        )


class SessionImageDataset(Dataset):
    """Flat image-label set from one session root (for gallery/probe eval)."""

    def __init__(
        self,
        session_root: str | Path,
        class_names: list[str] | None = None,
        class_to_idx: dict[str, int] | None = None,
        transform=None,
        image_size: int = 112,
        orig_only: bool = True,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
    ) -> None:
        self.root = Path(session_root)
        self.transform = transform or build_default_transform(image_size)
        self.image_size = image_size
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = (int(clahe_tile_grid[0]), int(clahe_tile_grid[1]))
        folders = sorted(
            d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        # Map normalized key → folder path (supports vein001_1 ↔ 001_1)
        key_to_dir = {_normalize_class_key(d.name): d for d in folders}

        if class_names is None:
            class_names = sorted(key_to_dir.keys())
        if class_to_idx is None:
            class_to_idx = {n: i for i, n in enumerate(class_names)}
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []
        for name in class_names:
            key = _normalize_class_key(name)
            if key not in key_to_dir:
                continue
            d = key_to_dir[key]
            paths = _list_images(d)
            if orig_only:
                paths = _prefer_orig(paths)
            label = class_to_idx[name]
            for p in paths:
                self.samples.append((p, label))
        if not self.samples:
            raise RuntimeError(f"No images under {session_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        if self.use_clahe:
            img = _pil_clahe(
                img,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid,
            )
        return self.transform(img), int(label)


__all__ = [
    "CrossSessionPoseDataset",
    "Session1OnlyPoseDataset",
    "SessionImageDataset",
    "DualSessionIndex",
    "_pil_clahe",
]
