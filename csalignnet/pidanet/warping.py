"""Differentiable geometric warping: rigid (θ,tx,ty), affine, and TPS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rigid_params_to_theta(theta: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    row0 = torch.stack([cos, -sin, tx], dim=-1)
    row1 = torch.stack([sin, cos, ty], dim=-1)
    return torch.stack([row0, row1], dim=1)


def affine_params_to_theta(params: torch.Tensor) -> torch.Tensor:
    return params.view(-1, 2, 3)


def warp_affine(x: torch.Tensor, theta_2x3: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid = F.affine_grid(theta_2x3, size=x.size(), align_corners=align_corners)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)


class TPSGrid(nn.Module):
    def __init__(self, grid_size: int = 3) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size
        xs = torch.linspace(-1.0, 1.0, grid_size)
        ys = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        src = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.register_buffer("src_points", src)

    @staticmethod
    def _phi(r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        return r2 * torch.log(r2.clamp_min(1e-6))

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n = self.num_points
        src = self.src_points.unsqueeze(0).expand(b, -1, -1)
        off = offsets.view(b, n, 2)
        dst = src + off
        device = x.device
        ones = torch.ones(b, n, 1, device=device, dtype=x.dtype)
        d = dst.unsqueeze(2) - dst.unsqueeze(1)
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        k = self._phi(r)
        p = torch.cat([ones, dst], dim=-1)
        z = torch.zeros(b, 3, 3, device=device, dtype=x.dtype)
        top = torch.cat([k, p], dim=-1)
        bot = torch.cat([p.transpose(1, 2), z], dim=-1)
        l = torch.cat([top, bot], dim=1)
        zeros = torch.zeros(b, 3, 2, device=device, dtype=x.dtype)
        rhs = torch.cat([src, zeros], dim=1)
        try:
            w_tps = torch.linalg.solve(l, rhs)
        except RuntimeError:
            eye = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device, dtype=x.dtype)
            grid = F.affine_grid(eye.unsqueeze(0).expand(b, -1, -1), size=x.size(), align_corners=True)
            return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=x.dtype),
            indexing="ij",
        )
        grid_pts = torch.stack([gx, gy], dim=-1).view(1, -1, 2).expand(b, -1, -1)
        diff = grid_pts.unsqueeze(2) - dst.unsqueeze(1)
        rr = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        u_k = self._phi(rr)
        u_p = torch.cat([torch.ones(b, h * w, 1, device=device, dtype=x.dtype), grid_pts], dim=-1)
        u = torch.cat([u_k, u_p], dim=-1)
        src_grid = torch.bmm(u, w_tps).view(b, h, w, 2)
        return F.grid_sample(x, src_grid, mode="bilinear", padding_mode="border", align_corners=True)


class DifferentiableWarper(nn.Module):
    def __init__(
        self,
        mode: str = "rt",
        tps_grid: int = 3,
        rot_scale: float = math.pi / 4,
        t_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.tps = TPSGrid(tps_grid) if mode == "tps" else None
        # rt bounds in normalized affine coords (image span ≈ [-1, 1])
        self.rot_scale = float(rot_scale)
        self.t_scale = float(t_scale)

    @property
    def num_params(self) -> int:
        if self.mode == "rt":
            return 3
        if self.mode == "affine":
            return 6
        if self.mode == "tps":
            assert self.tps is not None
            return self.tps.num_points * 2
        raise ValueError(self.mode)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "rt":
            th = torch.tanh(params[:, 0]) * self.rot_scale
            tx = torch.tanh(params[:, 1]) * self.t_scale
            ty = torch.tanh(params[:, 2]) * self.t_scale
            theta = rigid_params_to_theta(th, tx, ty)
            return warp_affine(x, theta), theta
        if self.mode == "affine":
            base = params.new_tensor([1, 0, 0, 0, 1, 0]).view(1, 6)
            raw = base + 0.1 * torch.tanh(params)
            theta = affine_params_to_theta(raw)
            return warp_affine(x, theta), theta
        if self.mode == "tps":
            assert self.tps is not None
            off = 0.15 * torch.tanh(params)
            warped = self.tps(x, off)
            eye = params.new_tensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3).expand(x.size(0), -1, -1)
            return warped, eye
        raise ValueError(self.mode)
