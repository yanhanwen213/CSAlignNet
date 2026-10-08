"""Angular-margin classification head (ArcFace / CosFace).

Deng et al., ArcFace, CVPR 2019:      cos(theta + m)
Wang et al., CosFace, CVPR 2018:      cos(theta) - s*m   (s=scale here)

Key properties for this project:
  * Replaces the plain-CE logits of the LoRaLin classifier during TRAINING only.
  * Inference path is untouched: eval still uses cosine(template) retrieval and
    the original LoRaLin logits, so Rank-1/EER gains come purely from a better
    trained backbone/head — zero extra inference parameters.
  * Feature/weights are L2-normalised, so the margin acts in ANGLE space:
    same-class samples are pushed to be angularly closer than margin m,
    which directly widens the same/different score gap -> EER drops.

Formula (ArcFace):
    logits_i = s * cos(theta_{y_i} + m)          for the true class
    logits_j = s * cos(theta_j)                  for other classes
with s = feature scale, m = angular margin (radians).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularMarginHead(nn.Module):
    """Plug-in replacement for `classifier(logits)` when computing CE.

    Usage:
        head = AngularMarginHead(embed_dim, num_classes, s=64.0, m=0.35,
                                 margin_type="cosface" | "arcface",
                                 easy_margin=True)
        logits_am = head(emb, labels)     # training-time (needs labels)
        logits_plain = head(emb)          # eval-time, = s * cos(theta), no margin

    The base weights W [num_classes, embed_dim] are the only parameters;
    gradient flows into the backbone through cos(theta).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.35,
        margin_type: str = "cosface",
        easy_margin: bool = True,
    ) -> None:
        super().__init__()
        if margin_type not in ("arcface", "cosface"):
            raise ValueError("margin_type must be 'arcface' or 'cosface'")
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)
        self.margin_type = margin_type
        self.easy_margin = bool(easy_margin)
        # class prototypes in angle space (same shape as ArcFace weights)
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cosine between sample and each class prototype
        emb_n = F.normalize(emb, p=2, dim=1)
        w_n = F.normalize(self.weight, p=2, dim=1)
        cosine = emb_n @ w_n.t()  # [B, C] in [-1, 1]
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            # eval / plain path: scaled cosine, no margin
            return self.s * cosine

        if self.margin_type == "arcface":
            theta = torch.acos(cosine)  # [B, C]
            if self.easy_margin:
                # only add margin when cos(theta) > cos(pi - 2m):
                # prevents huge/NaN gradients for grossly misaligned samples
                target = torch.where(
                    cosine > math.cos(math.pi - 2 * self.m),
                    torch.cos(theta + self.m),
                    cosine - self.m,
                )
            else:
                target = torch.cos(theta + self.m)
        else:  # cosface
            target = cosine - self.m

        # one-hot: true class uses margin-penalised cosine, others use cosine
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return self.s * logits

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_classes={self.num_classes}, "
            f"s={self.s}, m={self.m}, type={self.margin_type}, "
            f"easy_margin={self.easy_margin}"
        )


__all__ = ["AngularMarginHead"]
