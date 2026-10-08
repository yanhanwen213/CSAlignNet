"""
CSAlignNet: Alignment front-end + LRFVNet + Session-Invariant Head.

End-to-end:
  x -> AlignmentHead -> Warper(rt) -> LRFVNet -> SessionInvariantHead
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2] / "LRFVNet"
if _ROOT.is_dir() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrfvnet import LRFVNet  # noqa: E402

from .alignment import AlignmentHead
from .angular_margin import AngularMarginHead
from .domain_adv import DomainClassifier
from .session_head import SessionInvariantHead
from .warping import DifferentiableWarper


class CSAlignNet(nn.Module):
    """Cross-Session Alignment Network (lightweight e2e)."""

    def __init__(
        self,
        num_classes: int,
        warp_mode: str = "rt",
        tps_grid: int = 3,
        in_channels: int = 1,
        gamma: float = 0.5,
        dropout: float = 0.5,
        session_hidden: int = 256,
        session_out: int = 128,
        session_dropout: float = 0.1,
        t_scale: float = 0.5,
        rot_scale: float = math.pi / 4,
        use_domain_adv: bool = False,
        use_alignment: bool = True,
        use_session_head: bool = True,
        domain_hidden: int = 64,
        # ---- Feature-statistics style aug (train-time, no S2) ----
        style_aug: str = "none",
        style_p: float = 0.5,
        style_alpha: float = 0.1,
        # ---- Angular-margin classification (training-only) ----
        use_angular: bool = False,
        angular_type: str = "cosface",
        angular_s: float = 64.0,
        angular_m: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_alignment = bool(use_alignment)
        self.use_session_head = bool(use_session_head)
        self.use_domain_adv = bool(use_domain_adv) and self.use_session_head
        self.warper = DifferentiableWarper(
            mode=warp_mode,
            tps_grid=tps_grid,
            rot_scale=rot_scale,
            t_scale=t_scale,
        )
        self.align_head = (
            AlignmentHead(in_channels=in_channels, num_params=self.warper.num_params)
            if self.use_alignment
            else None
        )
        self.recognizer = LRFVNet(
            num_classes=num_classes,
            in_channels=in_channels,
            gamma=gamma,
            dropout=dropout,
            style_aug=style_aug,
            style_p=style_p,
            style_alpha=style_alpha,
        )
        emb_dim = self.recognizer.embed_dim
        self.feature_dim = session_out if self.use_session_head else emb_dim
        self.session_head = (
            SessionInvariantHead(
                in_dim=emb_dim,
                hidden_dim=session_hidden,
                out_dim=session_out,
                dropout=session_dropout,
            )
            if self.use_session_head
            else None
        )
        self.domain_clf = (
            DomainClassifier(in_dim=self.feature_dim, hidden_dim=domain_hidden)
            if self.use_domain_adv
            else None
        )
        self.use_angular = bool(use_angular)
        self.angular_head = None
        if self.use_angular:
            self.angular_head = AngularMarginHead(
                embed_dim=emb_dim,
                num_classes=num_classes,
                s=float(angular_s),
                m=float(angular_m),
                margin_type=str(angular_type),
                easy_margin=True,
            )

    def angular_logits(
        self, emb: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.angular_head is None:
            raise RuntimeError("Angular head disabled; set use_angular=True.")
        return self.angular_head(emb, labels)

    def align(self, x: torch.Tensor):
        if not self.use_alignment:
            b = x.size(0)
            eye = x.new_tensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3).expand(b, -1, -1)
            params = x.new_zeros((b, 3))
            pose_feat = x.new_zeros((b, 1))
            return x, params, pose_feat, eye
        params, pose_feat = self.align_head(x)
        aligned, theta = self.warper(x, params)
        return aligned, params, pose_feat, theta

    def retrieval_feature(self, emb: torch.Tensor, session_emb: torch.Tensor | None):
        if self.use_session_head:
            return session_emb
        return emb

    def domain_logits(self, session_emb: torch.Tensor, grl_lambda: float = 1.0):
        if self.domain_clf is None:
            raise RuntimeError("Domain classifier disabled; set use_domain_adv=True.")
        return self.domain_clf(session_emb, grl_lambda=grl_lambda)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        aligned, params, pose_feat, theta = self.align(x)
        logits, emb = self.recognizer(aligned, return_embedding=True)
        if self.use_session_head:
            session_emb = self.session_head(emb)
        else:
            session_emb = emb
        # Training uses angular logits when enabled; inference still uses
        # cosine retrieval on embeddings, so no behavioural change at eval.
        # `angular_logits(emb, labels)` is computed in the loss module.
        if return_aux:
            return {
                "logits": logits,
                "embedding": emb,
                "session_embedding": session_emb,
                "aligned": aligned,
                "pose_params": params,
                "pose_feat": pose_feat,
                "theta": theta,
            }
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def active_module_report(self) -> dict:
        return {
            "use_alignment": self.use_alignment,
            "use_session_head": self.use_session_head,
            "use_domain_adv": self.use_domain_adv,
        }
