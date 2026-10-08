"""Session-invariant projection head (Scheme A)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionInvariantHead(nn.Module):
    """
    Lightweight MLP that maps backbone embeddings to a session-stable space.

    Training pulls P(e_s1) and P(e_s2) together for the same identity.
    Inference matching / verification uses z = P(e).
    """

    def __init__(
        self,
        in_dim: int = 2704,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        z = self.net(embedding)
        return F.normalize(z, p=2, dim=1)
