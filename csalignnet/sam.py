# -*- coding: utf-8 -*-
"""Sharpness-Aware Minimization wrapper (Foret et al., ICLR 2021).

Correct Adam-compatible implementation: old weights are stored in a
private dict, NOT in base_optimizer.state, so Adam's exp_avg init
(len(state)==0 check) is not corrupted.

Usage:
    base = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    opt  = SAM(model.parameters(), base, rho=0.05, adaptive=False)
    loss.backward(); opt.first_step(zero_grad=True)
    loss2.backward(); opt.second_step(zero_grad=True)
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise ValueError("base_optimizer must be an Optimizer instance")
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for g in self.param_groups:
            g["rho"] = self.rho
            g["adaptive"] = self.adaptive
        self.defaults = self.base_optimizer.defaults
        self.state = self.base_optimizer.state
        self._old_p: dict[torch.Tensor, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                self._old_p[p] = p.data.clone()
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p in self._old_p:
                    p.data = self._old_p[p]
        self._old_p.clear()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = (torch.abs(p) if self.adaptive else 1.0) * p.grad
                norms.append(g.norm(p=2).to(p.device))
        if not norms:
            return torch.tensor(0.0)
        total = torch.norm(torch.stack([n.to(norms[0].device) for n in norms]), p=2)
        return total

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs):
        raise RuntimeError("SAM requires first_step/second_step; use them instead of step()")

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
