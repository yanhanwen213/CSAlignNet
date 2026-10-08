"""Losses for CSAlignNet (align + identity + cross-session)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_session_infonce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised InfoNCE between S1/S2 session embeddings.

    Positives: same identity across the two sessions (and same-id duplicates in-batch).
    Negatives: other identities in the batch.
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    b = z1.size(0)
    if b < 2:
        return z1.new_zeros(())

    # [2B, D], labels duplicated for both sessions
    z = torch.cat([z1, z2], dim=0)
    y = torch.cat([labels, labels], dim=0)
    sim = (z @ z.T) / temperature  # [2B, 2B]

    # mask self-comparisons
    self_mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    pos = y.unsqueeze(0) == y.unsqueeze(1)  # [2B, 2B]
    pos = pos & ~self_mask
    # rows with no positive (shouldn't happen for paired S1/S2) → skip
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return z1.new_zeros(())

    # log(sum_j exp(sim_ij)) for all j≠i
    log_den = torch.logsumexp(sim, dim=1)
    # log(sum_k∈pos exp(sim_ik))
    sim_pos = sim.masked_fill(~pos, -1e9)
    log_num = torch.logsumexp(sim_pos, dim=1)
    loss = (log_den - log_num)[valid].mean()
    return loss


class CSAlignLoss(nn.Module):
    """
    L = w_pose·CE(pose)+w_s1·CE(s1)+w_s2·CE(s2)
        + λ_tri·Tri(emb) + λ_sess·InfoNCE(z_a,z_b) + λ_reg·||p||²
        + λ_syn·InfoNCE(clean, pose)
        + λ_dom·CE(domain | GRL(z))   # optional domain-adversarial

    When the model has `use_angular=True`, the three CE terms on
    pose/s1/s2 are computed with AngularMarginHead (ArcFace/CosFace,
    training-only, zero extra inference params).
    """

    def __init__(
        self,
        margin: float = 0.5,
        lam_tri: float = 0.5,
        lam_sess: float = 1.0,
        lam_reg: float = 0.01,
        lam_syn: float = 0.0,
        lam_dom: float = 0.0,
        w_pose: float = 1.0,
        w_s1: float = 1.0,
        w_s2: float = 1.0,
        sess_temperature: float = 0.07,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.lam_tri = lam_tri
        self.lam_sess = lam_sess
        self.lam_reg = lam_reg
        self.lam_syn = lam_syn
        self.lam_dom = lam_dom
        self.w_pose = w_pose
        self.w_s1 = w_s1
        self.w_s2 = w_s2
        self.sess_temperature = sess_temperature
        self.grl_lambda = grl_lambda
        self.ce = nn.CrossEntropyLoss()

    def _ce_or_angular(
        self,
        out: dict,
        labels: torch.Tensor,
        model: nn.Module | None,
    ) -> torch.Tensor:
        if (
            model is not None
            and getattr(model, "use_angular", False)
            and getattr(model, "angular_head", None) is not None
        ):
            # Angular margin on the backbone embedding (training path)
            return self.ce(model.angular_logits(out["embedding"], labels), labels)
        return self.ce(out["logits"], labels)

    def triplet(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)
        n = F.normalize(n, p=2, dim=1)
        d_ap = torch.sum((a - p) ** 2, dim=1)
        d_an = torch.sum((a - n) ** 2, dim=1)
        return F.relu(d_ap - d_an + self.margin).mean()

    def forward(
        self,
        out_pose: dict,
        labels: torch.Tensor,
        out_s1: dict | None = None,
        out_s2: dict | None = None,
        out_p: dict | None = None,
        out_n: dict | None = None,
        out_clean: dict | None = None,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_id = self.w_pose * self._ce_or_angular(out_pose, labels, model)
        if out_s1 is not None:
            loss_id = loss_id + self.w_s1 * self._ce_or_angular(out_s1, labels, model)
        if out_s2 is not None:
            loss_id = loss_id + self.w_s2 * self._ce_or_angular(out_s2, labels, model)
        loss_reg = (out_pose["pose_params"] ** 2).mean()

        loss_tri = labels.new_zeros(())
        if out_p is not None and out_n is not None:
            loss_tri = self.triplet(
                out_pose["embedding"], out_p["embedding"], out_n["embedding"]
            )

        loss_sess = labels.new_zeros(())
        use_sess_head = True if model is None else bool(getattr(model, "use_session_head", True))
        if use_sess_head and out_s1 is not None and out_s2 is not None:
            loss_sess = cross_session_infonce(
                out_s1["session_embedding"],
                out_s2["session_embedding"],
                labels,
                temperature=self.sess_temperature,
            )

        loss_syn = labels.new_zeros(())
        if self.lam_syn > 0 and use_sess_head and out_clean is not None:
            # pose view vs clean S1: same identity positives in-batch
            loss_syn = cross_session_infonce(
                out_clean["session_embedding"],
                out_pose["session_embedding"],
                labels,
                temperature=self.sess_temperature,
            )

        # Domain-adversarial: confuse clean(a)=0 vs pseudo/S2(b)=1 on session emb.
        loss_dom = labels.new_zeros(())
        if (
            self.lam_dom > 0
            and model is not None
            and use_sess_head
            and getattr(model, "use_domain_adv", False)
            and out_s1 is not None
            and out_s2 is not None
        ):
            z = torch.cat(
                [out_s1["session_embedding"], out_s2["session_embedding"]], dim=0
            )
            dom_y = torch.cat(
                [
                    labels.new_zeros(labels.size(0), dtype=torch.long),
                    labels.new_ones(labels.size(0), dtype=torch.long),
                ],
                dim=0,
            )
            d_logits = model.domain_logits(z, grl_lambda=self.grl_lambda)
            loss_dom = self.ce(d_logits, dom_y)

        loss = (
            loss_id
            + self.lam_tri * loss_tri
            + self.lam_sess * loss_sess
            + self.lam_reg * loss_reg
            + self.lam_syn * loss_syn
            + self.lam_dom * loss_dom
        )
        stats = {
            "loss": float(loss.detach()),
            "id": float(loss_id.detach()),
            "tri": float(loss_tri.detach()),
            "sess": float(loss_sess.detach()),
            "reg": float(loss_reg.detach()),
            "syn": float(loss_syn.detach()),
            "dom": float(loss_dom.detach()),
        }
        return loss, stats
