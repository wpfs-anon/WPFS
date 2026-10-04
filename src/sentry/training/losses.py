from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "lambda_a",
    "token_loss",
    "behaviour_loss",
    "stage_a_loss",
    "distillation_loss",
    "margin_loss",
    "sensitivity_loss",
    "stage_b_loss",
    "curriculum",
    "multi_exit_loss",
    "StageBLosses",
]


def lambda_a(u: float, lambda_max: float) -> float:
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"training fraction u must lie in [0,1], got {u}")
    return lambda_max * u


def token_loss(z_shal: Tensor, z_full: Tensor) -> Tensor:
    if z_shal.shape != z_full.shape:
        raise ValueError(
            f"token sequences must match: {tuple(z_shal.shape)} vs {tuple(z_full.shape)}"
        )
    cos = F.cosine_similarity(z_shal, z_full.detach(), dim=-1)
    return 1.0 - cos.mean()


def behaviour_loss(
    v_shal: Tensor, v_full: Tensor, channels: Optional[Sequence[int]] = None
) -> Tensor:
    if v_shal.shape != v_full.shape:
        raise ValueError(
            f"velocities must match: {tuple(v_shal.shape)} vs {tuple(v_full.shape)}"
        )
    d = (v_shal - v_full.detach()).pow(2)
    if channels is not None:
        idx = torch.as_tensor(tuple(channels), dtype=torch.long, device=d.device)
        d = d.index_select(-1, idx)
    return d.mean()


def gripper_sign_loss(
    R_shal: Tensor,
    R_full: Tensor,
    grip: int,
    margin: float = 0.1,
    dead_zone: float = 1e-3,
) -> Tensor:
    s = R_full.detach()[..., grip]
    t = R_shal[..., grip]
    live = s.abs() > dead_zone
    if not bool(live.any()):
        return R_shal.new_zeros(())
    hinge = F.relu(margin - t * torch.sign(s))
    return (hinge * live).sum() / live.sum()


def stage_a_loss(
    z_shal: Tensor,
    z_full: Tensor,
    v_shal: Tensor,
    v_full: Tensor,
    u: float,
    lambda_max: float,
    channels: Optional[Sequence[int]] = None,
    R_shal: Optional[Tensor] = None,
    R_full: Optional[Tensor] = None,
    grip: Optional[int] = None,
    lambda_grip: float = 0.0,
    grip_margin: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    l_tok = token_loss(z_shal, z_full)
    l_beh = behaviour_loss(v_shal, v_full, channels)
    lam = lambda_a(u, lambda_max)
    loss = l_tok + lam * l_beh

    parts = {
        "L_tok": float(l_tok.detach()),
        "L_beh": float(l_beh.detach()),
        "lambda_A": lam,
    }

    if lambda_grip and R_shal is not None and R_full is not None and grip is not None:
        l_grip = gripper_sign_loss(R_shal, R_full, grip, margin=grip_margin)
        loss = loss + lambda_grip * l_grip
        parts["L_grip"] = float(l_grip.detach())
        with torch.no_grad():
            flip = (
                (R_shal[..., grip] > 0) != (R_full[..., grip] > 0)
            ).float().mean()
        parts["grip_flip"] = float(flip)

    parts["loss"] = float(loss.detach())
    return loss, parts


def distillation_loss(v_shal: Tensor, v_full: Tensor) -> Tensor:
    if v_shal.shape != v_full.shape:
        raise ValueError(
            f"velocities must match: {tuple(v_shal.shape)} vs {tuple(v_full.shape)}"
        )
    return (v_shal - v_full.detach()).pow(2).mean()


def margin_loss(d: Tensor, h_star: int, H_k: int, m: float) -> Tensor:
    if d.ndim != 1:
        raise ValueError(f"d must be (H_k,), got {tuple(d.shape)}")
    if not (0 <= h_star <= H_k):
        raise ValueError(f"h_star={h_star} outside [0, {H_k}]")
    if H_k > d.shape[0]:
        raise ValueError(f"H_k={H_k} exceeds available positions {d.shape[0]}")

    loss = d.new_zeros(())

    if h_star > 0:
        push_down = F.relu(d[:h_star] - (1.0 - m))
        loss = loss + push_down.sum() / h_star

    if h_star < H_k:
        loss = loss + F.relu((1.0 + m) - d[h_star])

    return loss


def sensitivity_loss(R_pos: Tensor, R_neg: Tensor, eta: float) -> Tensor:
    if R_pos.shape != R_neg.shape:
        raise ValueError(
            f"matched pair must match: {tuple(R_pos.shape)} vs {tuple(R_neg.shape)}"
        )
    separation = torch.linalg.vector_norm((R_pos - R_neg).flatten(1), ord=2, dim=-1)
    return F.relu(eta - separation).mean()


class StageBLosses(dict):

    @property
    def collapse_diagnostic(self) -> float:
        return self["separation"]


def stage_b_loss(
    v_shal: Tensor,
    v_full: Tensor,
    d: Tensor,
    h_star: Tensor,
    H_k: Tensor,
    R_pos: Optional[Tensor],
    R_neg: Optional[Tensor],
    lambda_m: float,
    lambda_s: float,
    m: float,
    eta: float,
) -> tuple[Tensor, StageBLosses]:
    l_dist = distillation_loss(v_shal, v_full)

    l_marg = d.new_zeros(())
    B = d.shape[0]
    for i in range(B):
        l_marg = l_marg + margin_loss(d[i], int(h_star[i]), int(H_k[i]), m)
    l_marg = l_marg / max(B, 1)

    if R_pos is not None and R_neg is not None:
        l_sens = sensitivity_loss(R_pos, R_neg, eta)
        separation = float(
            torch.linalg.vector_norm((R_pos - R_neg).flatten(1), ord=2, dim=-1)
            .mean()
            .detach()
        )
    else:
        l_sens = d.new_zeros(())
        separation = float("nan")

    loss = l_dist + lambda_m * l_marg + lambda_s * l_sens
    parts = StageBLosses(
        L_dist=float(l_dist.detach()),
        L_marg=float(l_marg.detach()),
        L_sens=float(l_sens.detach()),
        separation=separation,
        loss=float(loss.detach()),
    )
    return loss, parts


def curriculum(u: float, enable_at: Sequence[float]) -> list[float]:
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"training fraction u must lie in [0,1], got {u}")
    return [1.0 if u >= uj else 0.0 for uj in enable_at]


def multi_exit_loss(
    per_rung: Sequence[Tensor],
    weights: Sequence[float],
    u: float,
    enable_at: Sequence[float],
) -> tuple[Tensor, dict[str, float]]:
    if not (len(per_rung) == len(weights) == len(enable_at)):
        raise ValueError(
            f"parallel sequences disagree: {len(per_rung)} losses, "
            f"{len(weights)} weights, {len(enable_at)} curriculum points"
        )
    if not per_rung:
        raise ValueError("need at least one rung")

    c = curriculum(u, enable_at)
    total = per_rung[0].new_zeros(())
    parts: dict[str, float] = {}
    for j, (loss_j, w_j, c_j) in enumerate(zip(per_rung, weights, c)):
        total = total + w_j * c_j * loss_j
        parts[f"L_rung{j}"] = float(loss_j.detach())
        parts[f"c_rung{j}"] = c_j
    parts["active_rungs"] = sum(c)
    parts["loss"] = float(total.detach())
    return total, parts
