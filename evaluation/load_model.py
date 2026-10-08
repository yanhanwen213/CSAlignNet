"""Load LRFVNet, CSAlignNet, B3, B4, or B5 checkpoints for evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
_CSALIGNNET = REPO / "csalignnet"
_LRFV = Path(__import__("os").environ.get("LRFVNET_PATH", str(REPO.parent / "LRFVNet")))
for p in (_CSALIGNNET, _LRFV):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lrfvnet import LRFVNet  # noqa: E402  (external LRFVNet dependency)
from pidanet import CSAlignNet  # noqa: E402  (csalignnet package)


def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    state = ckpt["model"]
    return ckpt, args, state


def build_lrfvnet(state: dict, args: dict, num_classes: int | None, device: torch.device) -> LRFVNet:
    n_cls = num_classes or args.get("num_classes")
    if n_cls is None:
        # LoRaLin up.weight: [out, rank]
        key = "classifier.up.weight" if "classifier.up.weight" in state else "classifier.weight"
        n_cls = state[key].shape[0]
    model = LRFVNet(
        num_classes=n_cls,
        gamma=args.get("gamma", 0.5),
        dropout=args.get("dropout", 0.5),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_csalign(state: dict, args: dict, num_classes: int | None, device: torch.device) -> CSAlignNet:
    n_cls = num_classes or args.get("num_classes")
    if n_cls is None:
        key = "recognizer.classifier.up.weight"
        if key not in state:
            key = "recognizer.classifier.weight"
        n_cls = state[key].shape[0]
    model = CSAlignNet(
        num_classes=n_cls,
        warp_mode=args.get("warp_mode", "rt"),
        tps_grid=args.get("tps_grid", 3),
        gamma=args.get("gamma", 0.5),
        dropout=args.get("dropout", 0.5),
        t_scale=args.get("t_scale", 0.5),
        session_hidden=args.get("session_hidden", 256),
        session_out=args.get("session_out", 128),
        style_aug=args.get("style_aug", "none"),
        style_p=args.get("style_p", 0.5),
        style_alpha=args.get("style_alpha", 0.1),
        use_angular=bool(args.get("angular", False)),
        angular_type=args.get("angular_type", "cosface"),
        angular_s=args.get("angular_s", 64.0),
        angular_m=args.get("angular_m", 0.35),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[build_csalign] missing keys: {missing}")
    if unexpected:
        print(f"[build_csalign] unexpected keys: {unexpected}")
    model.to(device).eval()
    return model


def build_b3(state: dict, args: dict, num_classes: int | None, device: torch.device):
    from b3 import B3Pipeline  # noqa: WPS433

    n_cls = num_classes or args.get("num_classes")
    if n_cls is None:
        key = "recognizer.classifier.up.weight"
        if key not in state:
            key = "recognizer.classifier.weight"
        n_cls = state[key].shape[0]
    model = B3Pipeline(
        num_classes=n_cls,
        angle_scale=args.get("angle_scale", 35.0),
        shift_scale=args.get("shift_scale", 40.0),
        gamma=args.get("gamma", 0.5),
        dropout=args.get("dropout", 0.5),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_b4(state: dict, args: dict, num_classes: int | None, device: torch.device):
    from b4 import B4STN  # noqa: WPS433

    n_cls = num_classes or args.get("num_classes")
    if n_cls is None:
        key = "recognizer.classifier.up.weight"
        if key not in state:
            key = "recognizer.classifier.weight"
        n_cls = state[key].shape[0]
    model = B4STN(
        num_classes=n_cls,
        warp_mode=args.get("warp_mode", "rt"),
        tps_grid=args.get("tps_grid", 3),
        gamma=args.get("gamma", 0.5),
        dropout=args.get("dropout", 0.5),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_b5(state: dict, args: dict, num_classes: int | None, device: torch.device):
    from b5 import MobileNetV3SmallFV  # noqa: WPS433

    n_cls = num_classes or args.get("num_classes")
    if n_cls is None:
        n_cls = state["classifier.weight"].shape[0]
    model = MobileNetV3SmallFV(
        num_classes=n_cls,
        in_channels=1,
        dropout=args.get("dropout", 0.5),
        pretrained=False,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_model(
    ckpt_path: str | Path,
    model_type: str,
    device: torch.device,
    num_classes: int | None = None,
):
    ckpt, args, state = load_checkpoint(ckpt_path, device)
    model_type = model_type.lower()
    if model_type in ("lrfv", "lrfvnet", "lrf"):
        model = build_lrfvnet(state, args, num_classes, device)
    elif model_type in ("pida", "pidanet", "csalign", "cs_align", "cs"):
        model = build_csalign(state, args, num_classes, device)
    elif model_type in ("b3", "twostage", "pose_lrfv"):
        model = build_b3(state, args, num_classes, device)
    elif model_type in ("b4", "stn", "stn_lrfv"):
        model = build_b4(state, args, num_classes, device)
    elif model_type in ("b5", "mobilenet", "mobilenetv3", "mnv3"):
        model = build_b5(state, args, num_classes, device)
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    meta = {
        "epoch": ckpt.get("epoch"),
        "acc": ckpt.get("acc"),
        "f1": ckpt.get("f1"),
        "eer": ckpt.get("eer"),
        "fold": ckpt.get("fold"),
        "args": args,
    }
    return model, meta
