"""Feature-statistics style augmentations (train-time only, no S2).

MixStyle:  Zhou et al., ICLR 2021  (mix instance mean/var across batch)
DSU:       Li et al., ICLR 2022    (sample new mean/var from Gaussian)
pAdaIN:    Huang & Belongie, ICCV 2017 / trivial stochastic IN statistics swap

All modules are identity at eval (self.training==False or p==0).
They operate on feature maps (B,C,H,W) and keep spatial structure unchanged
so vein topology is preserved while illumination/style statistics are diversified.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle after a conv block.

    With prob `p`, shuffle batch and convex-mix channel-wise mean/std:
      beta = lam*mu + (1-lam)*mu_perm ; gamma analog.  lam ~ Beta(alpha,alpha)
    Ref: Zhou et al., Domain Generalization with MixStyle, ICLR 2021.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        # Beta is sampled on the fly; no learnable params
        if self.alpha <= 0:
            raise ValueError("MixStyle alpha must be >0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or random.random() > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        lam = torch.distributions.Beta(
            torch.tensor(self.alpha), torch.tensor(self.alpha)
        ).sample().item()
        # clamp away from 0/1 to avoid degenerate swap
        lam = float(max(0.05, min(0.95, lam)))

        perm = torch.randperm(b, device=x.device)
        # avoid identity perm when b==2 (rare, not critical)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)
        return x_norm * sig_mix + mu_mix

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"


class DSU(nn.Module):
    """Distribution Uncertainty (DSU) - stochastic style hallucination.

    Perturb per-instance mean/var with Gaussian noise whose variance is
    estimated from the batch distribution of that statistic.
    Ref: Li et al., Uncertainty Modeling for Out-of-Distribution Generalization, ICLR 2022.
    """

    def __init__(self, p: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or random.random() > self.p:
            return x
        b, c = x.size(0), x.size(1)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3])  # [B,C]
        var = x.var(dim=[2, 3], unbiased=False)
        sig = (var + self.eps).sqrt()  # [B,C]

        # batch-wise variance of the statistics
        mu_var = mu.var(dim=0, keepdim=True)  # [1,C]
        sig_var = sig.var(dim=0, keepdim=True)

        # sample new statistics
        mu_eps = torch.randn_like(mu) * (mu_var + self.eps).sqrt()
        sig_eps = torch.randn_like(sig) * (sig_var + self.eps).sqrt()

        mu_new = mu + mu_eps
        sig_new = sig + sig_eps
        sig_new = sig_new.clamp_min(self.eps)

        # normalize then re-style with sampled stats
        mu_b = mu.view(b, c, 1, 1)
        sig_b = sig.view(b, c, 1, 1)
        mu_n = mu_new.view(b, c, 1, 1)
        sig_n = sig_new.view(b, c, 1, 1)
        x_norm = (x - mu_b) / sig_b
        return x_norm * sig_n + mu_n

    def extra_repr(self) -> str:
        return f"p={self.p}"


class pAdaIN(nn.Module):
    """Stochastic IN swap (pAdaIN): with prob p, replace each sample's
    mean/var with another random sample's in the batch.
    Lightweight baseline; stronger variants are MixStyle/DSU.
    """

    def __init__(self, p: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or random.random() > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = x.var(dim=[2, 3], keepdim=True, unbiased=False).add(self.eps).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(b, device=x.device)
        return x_norm * sig[perm] + mu[perm]

    def extra_repr(self) -> str:
        return f"p={self.p}"


__all__ = ["MixStyle", "DSU", "pAdaIN"]
