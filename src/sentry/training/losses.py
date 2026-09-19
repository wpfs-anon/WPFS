r"""Training objectives -- equations 13 to 16 (SS2.5).

Pure functions, deliberately separated from the training loops, because the
subtle parts of this recipe are all *in* the objectives:

- eq. 13's behavioural term must dominate the token term by the end of
  training, not the other way round;
- eq. 14 distils the target's **verification response**, not its policy;
- eq. 15's second term is *omitted* when ``h_star == H_k``;
- eq. 16 exists solely to exclude a degenerate optimum that would otherwise
  score perfectly on eq. 14.

Each of those is a one-line mistake with no visible symptom during training --
the loss goes down either way -- which is why they are unit-tested rather than
merely written down.
"""

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


# --------------------------------------------------------------------------
# Stage A -- equation 13
# --------------------------------------------------------------------------


def lambda_a(u: float, lambda_max: float) -> float:
    """``lambda_A(u)``, annealed from 0 to ``lambda_max`` over training fraction ``u``.

    SS2.5.1: "the token term is a fast warm start that puts the truncated
    encoder in the right region, after which the behavioural term takes over
    and **is the one that matters**."

    Annealing the wrong way -- starting at ``lambda_max`` and decaying -- trains
    the encoder to match intermediate tokens it was never required to match,
    which SS2.5.1 calls "both unnecessary and, at ``E_V << L_V``, likely
    infeasible".
    """
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"training fraction u must lie in [0,1], got {u}")
    return lambda_max * u


def token_loss(z_shal: Tensor, z_full: Tensor) -> Tensor:
    """``L_tok = 1 - cos(z_shal, z_full)``, averaged over tokens.

    Cosine rather than L2 because what matters is direction in representation
    space, not scale: the truncated encoder need only "preserve the
    information the backbone uses".
    """
    if z_shal.shape != z_full.shape:
        raise ValueError(
            f"token sequences must match: {tuple(z_shal.shape)} vs {tuple(z_full.shape)}"
        )
    cos = F.cosine_similarity(z_shal, z_full.detach(), dim=-1)
    return 1.0 - cos.mean()


def behaviour_loss(
    v_shal: Tensor, v_full: Tensor, channels: Optional[Sequence[int]] = None
) -> Tensor:
    r""":math:`\|v_\theta(A^\tau,\tau\mid z^{shal}) - sg[v_\theta(A^\tau,\tau\mid z^{full})]\|_2^2`.

    Both branches run the **frozen full-depth backbone**; only the supplied
    visual tokens differ.  Stage A truncates the encoder, never the backbone.

    ``channels`` restricts the mean to the action dimensions that carry
    information -- see :attr:`sentry.core.types.ChannelSpec.real_channels`.
    Equation 13 as written averages over all ``d_a``, and on ``pi0_libero`` that
    is 7 real channels among 32, so 25 of them contribute a zero both branches
    produce for free.  The measured dilution is ``4.46x``.  Passing ``None``
    reproduces the paper's literal reading.
    """
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
    r"""Keep the shallow branch's gripper on the same side of zero as the target.

    The gripper is "discrete in intent" (SS3.4.3) and the acceptance rule tests
    it by **sign**, not by distance -- yet eq. 13 scores it with the same squared
    error as a translation channel, so nothing in Stage A's objective protects
    the one property the check operator actually reads.

    That gap is measurable.  With Stage A trained to a 96% reduction in
    ``L_beh``, the reconstructed gripper still disagreed in sign with the
    full-depth branch on **11% of chunk positions** -- and a sign flip is opening
    the hand instead of closing it, which fails the grasp no matter how small its
    contribution to a mean square.

    The hinge asks for agreement *with margin*: where the target is
    :math:`s = R^{full}_{grip}` and the student is :math:`t = R^{shal}_{grip}`,

    .. math::
        \mathcal{L}_{grip} = \big[\,m - t \cdot \mathrm{sign}(s)\,\big]_+

    so a student that merely creeps over zero is still penalised.  Positions
    where the target itself sits inside ``dead_zone`` of zero carry no intent to
    preserve and are excluded rather than given an arbitrary sign to chase.
    """
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
    r"""Equation 13, plus the two terms measurement showed it needs.

    .. math::
        \mathcal{L}_A = \mathcal{L}_{tok}
                      + \lambda_A(u)\,\mathcal{L}_{beh}
                      + \lambda_{grip}\,\mathcal{L}_{grip}

    ``channels`` and the gripper term default to off, so the literal reading of
    the paper is still reachable; both are switched on by supplying a
    :class:`ChannelSpec`'s fields.  Neither changes what Stage A *trains*
    (``Delta_V`` alone, backbone frozen) -- they change what it is *scored* on,
    which is where the objective and task success came apart.

    No position weighting: the executed prefix and the discarded tail were
    measured at 0.188 and 0.171 endpoint error respectively, near enough that
    weighting them differently would be a knob without a reason.
    """
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


# --------------------------------------------------------------------------
# Stage B -- equations 14, 15, 16
# --------------------------------------------------------------------------


def distillation_loss(v_shal: Tensor, v_full: Tensor) -> Tensor:
    r"""Equation 14: :math:`\mathcal{L}_{dist} = \|v^{shal} - sg[v^{full}]\|_2^2`.

    SS2.5.2: "This term, and **not a regression onto demonstration actions**,
    is what makes the shallow check an approximation of the gold-standard check
    rather than of the policy."

    The distinction is the whole design.  Regressing onto demonstrations would
    train a shallow *policy*; distilling the full-depth velocity **on the very
    inputs it will be queried on** trains a shallow *verifier*.
    """
    if v_shal.shape != v_full.shape:
        raise ValueError(
            f"velocities must match: {tuple(v_shal.shape)} vs {tuple(v_full.shape)}"
        )
    return (v_shal - v_full.detach()).pow(2).mean()


def margin_loss(d: Tensor, h_star: int, H_k: int, m: float) -> Tensor:
    r"""Equation 15 -- the decision margin, for one sample.

    .. math::
        \mathcal{L}_{marg} = \frac{1}{h^\star}\sum_{h<h^\star}
            \big[d_h - (1-m)\big]_+ \;+\; \big[(1+m) - d_{h^\star}\big]_+

    Args:
        d: ``(H_k,)`` normalised distances from ``R^{shal}`` (eq. 10),
           already reduced over ``T``.
        h_star: ground-truth first invalid position.  ``H_k`` for positives;
            ``h_0`` for N4; from a full-depth evaluation of eq. 5 for N1-N3.
        H_k: number of real entries.
        m: hinge margin.

    SS2.5.2: "Regression accuracy on ``v`` does not by itself produce a good
    decision, because the decision depends on a **thresholded functional** of
    ``v``."  So the acceptance statistic is optimised directly: valid positions
    are pushed comfortably below threshold, and the first invalid position
    comfortably above it.

    Two edge cases the equation implies but does not spell out:

    - **``h_star == H_k``** (a fully live plan): "The second term is omitted"
      -- there is no invalid position to push up.  Including it would train the
      model to reject the padded tail, which eq. 11 never even evaluates.
    - **``h_star == 0``** (position 0 already invalid): the sum is empty and
      ``1/h_star`` is undefined.  Only the push-up term applies.
    """
    if d.ndim != 1:
        raise ValueError(f"d must be (H_k,), got {tuple(d.shape)}")
    if not (0 <= h_star <= H_k):
        raise ValueError(f"h_star={h_star} outside [0, {H_k}]")
    if H_k > d.shape[0]:
        raise ValueError(f"H_k={H_k} exceeds available positions {d.shape[0]}")

    loss = d.new_zeros(())

    if h_star > 0:
        # Push valid positions comfortably BELOW threshold.
        push_down = F.relu(d[:h_star] - (1.0 - m))
        loss = loss + push_down.sum() / h_star

    if h_star < H_k:
        # Push the first invalid position comfortably ABOVE threshold.
        loss = loss + F.relu((1.0 + m) - d[h_star])

    return loss


def sensitivity_loss(R_pos: Tensor, R_neg: Tensor, eta: float) -> Tensor:
    r"""Equation 16 -- the anti-collapse regulariser.

    .. math::
        \mathcal{L}_{sens} = \big[\eta - \|R^{shal}(o^\star_+)
                             - R^{shal}(o^\star_-)\|_2\big]_+

    SS2.5.2 names the failure precisely: "A degenerate optimum of equation 14
    alone is a shallow mode that **ignores** ``o*`` and reproduces ``A_hat`` by
    copying the interpolant, which yields ``d_h ~ 0`` everywhere and a verifier
    that **accepts unconditionally**."

    That optimum scores *well* on eq. 14 whenever the full-depth teacher also
    roughly returns the candidate, so nothing else in the objective excludes
    it.  ``R_pos`` and ``R_neg`` must come from a **matched pair sharing the
    same** ``A_hat``, ``tau`` and ``eps`` -- if they differ in anything but the
    observation, this term measures noise instead of observation-sensitivity
    and the regulariser is silently inert.
    """
    if R_pos.shape != R_neg.shape:
        raise ValueError(
            f"matched pair must match: {tuple(R_pos.shape)} vs {tuple(R_neg.shape)}"
        )
    separation = torch.linalg.vector_norm((R_pos - R_neg).flatten(1), ord=2, dim=-1)
    return F.relu(eta - separation).mean()


class StageBLosses(dict):
    """Loss parts plus the collapse diagnostic, for logging."""

    @property
    def collapse_diagnostic(self) -> float:
        """``||R_shal(o*_+) - R_shal(o*_-)||``.

        SS2.5.2: "We report [this] during training as the collapse
        diagnostic."  It should stay comfortably above ``eta``; drifting toward
        zero means the verifier is learning to ignore its visual input, and the
        acceptance rate will look excellent right up until it is useless.
        """
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
    """``L_B = L_dist + lambda_m * L_marg + lambda_s * L_sens``.

    Args:
        v_shal, v_full: ``(B, H, d_a)``.
        d: ``(B, H)`` normalised distances, already reduced over ``T``.
        h_star, H_k: ``(B,)`` integer tensors.
        R_pos, R_neg: matched-pair reconstructions for eq. 16, or ``None`` to
            skip the term (e.g. a batch with no pairs available).
    """
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


# --------------------------------------------------------------------------
# Multi-exit aggregation -- equation 18
# --------------------------------------------------------------------------


def curriculum(u: float, enable_at: Sequence[float]) -> list[float]:
    r"""``c_j(u) = 1[u >= u_j]`` -- the binary curriculum of SS3.5.2.

    Args:
        u: training fraction in ``[0, 1]``.
        enable_at: ``u_j`` per rung, ordered ``u_1 > ... > u_J = 0``.

    "Rungs are enabled progressively... so training begins with the deepest rung
    alone and admits shallower rungs as the trunk adapter stabilises.  Without
    this, the shallow rungs -- whose targets are hardest -- dominate the gradient
    early and destabilise a trunk that is shared with every other rung."

    Note the ordering: index 0 is the *shallowest* rung and carries the largest
    ``u_j``, so it switches on last.  Reversing that trains in exactly the order
    the paper says destabilises the trunk, and the loss curve looks fine either
    way.
    """
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"training fraction u must lie in [0,1], got {u}")
    return [1.0 if u >= uj else 0.0 for uj in enable_at]


def multi_exit_loss(
    per_rung: Sequence[Tensor],
    weights: Sequence[float],
    u: float,
    enable_at: Sequence[float],
) -> tuple[Tensor, dict[str, float]]:
    r"""Equation 18: :math:`\mathcal{L}_B(u) = \sum_j w_j\, c_j(u)\, \mathcal{L}^{(j)}`.

    The three sequences are parallel and rung-ordered: ``per_rung[j]`` is the
    loss read out at :math:`E^{(j)}`, ``weights[j]`` is :math:`w_j`, and
    ``enable_at[j]`` is :math:`u_j`.

    Deliberately **not** renormalised by the active weights.  ``c_j`` gating a
    rung off is meant to remove its gradient, not to redistribute it: dividing by
    ``sum(w_j c_j)`` would silently inflate the deepest rung's learning rate
    early in training, which is the phase the curriculum exists to keep calm.
    The effective step size therefore *grows* as rungs switch on, which is the
    intended shape.
    """
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
