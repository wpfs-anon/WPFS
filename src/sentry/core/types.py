"""Core data types for SENTRY.

Nothing in this module may import a concrete model.  These types are the
vocabulary shared by the check operator (SS2.4), the depth cascade (SS2.6),
calibration (SS2.4.4) and the cost model (SS2.7).

Paper reference: "Beyond Action Matching" (ICLR 2026 submission), Section 2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "DepthRung",
    "Observation",
    "ChannelSpec",
    "CheckResult",
    "Thresholds",
]


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthRung:
    """One rung of the depth cascade: the pair ``(E_V, E_B)``.

    Paper defect **D2**: Table 2 gives a scalar ``E_V = 8`` and Algorithm 1's
    ``CHECK(A_hat, o_t, E_V, E_B^(j))`` takes it as a scalar, yet Table 1's
    rung 3 uses ``E_V = 14/27``.  Modelling a rung as a *pair* subsumes both
    readings, so we never have to choose.

    Note that a rung carries a single ``E_B`` for the whole backbone.  Per SS2.9
    ("Depth is shared"), attention is joint at every layer, so action tokens at
    layer ``l`` attend to the visual/language keys and values of layer ``l``;
    the perceptual and denoising pathways therefore *cannot* be truncated
    independently.  See :func:`assert_shared_depth`.
    """

    E_V: int
    E_B: int

    def __post_init__(self) -> None:
        if self.E_V < 1:
            raise ValueError(f"E_V must be >= 1, got {self.E_V}")
        if self.E_B < 1:
            raise ValueError(f"E_B must be >= 1, got {self.E_B}")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"(E_V={self.E_V}, E_B={self.E_B})"

    @property
    def key(self) -> tuple[int, int]:
        return (self.E_V, self.E_B)


def assert_shared_depth(e_perception: int, e_denoising: int) -> None:
    """Reject "shallow backbone, deep denoiser" configurations.

    SS2.9, *Depth is shared*: a single ``E_B`` governs both pathways, and
    configurations that truncate them independently "are not expressible
    without modifying the architecture".  We make that a construction-time
    error rather than a silently wrong model.
    """
    if e_perception != e_denoising:
        raise ValueError(
            "Perception and denoising depth must be identical: attention is "
            "joint at every layer, so skipping layer l skips both the VLM "
            "expert and the action expert.  A single E_B governs both "
            f"(got perception={e_perception}, denoising={e_denoising}).  "
            "See Section 2.9, 'Depth is shared'."
        )


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """``o_t = [I^1_t, ..., I^n_t, l_t, s_t]`` (SS2.1).

    ``t`` is carried explicitly so the execution loop can *assert* that the
    conditioning it feeds the verifier is fresh.  Algorithm 1 line 12 writes
    ``CHECK(A_hat, o_t, ...)``, but the observation must be re-read from the
    environment at every check iteration -- recomputing conditioning from the
    *current* images is exactly what Corollary 3 requires.  A stale ``o``
    would silently reduce SENTRY to the Proposition 2 straw man.
    """

    images: Tensor
    """``(n_cam, C, H, W)`` exteroceptive input."""

    language: Tensor
    """``(L_lang,)`` instruction token ids (or a pooled embedding)."""

    state: Tensor
    """``(d_state,)`` proprioception ``s_t``."""

    t: int
    """Environment timestep at which this observation was taken."""

    def with_state(self, state: Tensor, t: int) -> "Observation":
        """Refresh *only* proprioception, keeping the cached images/language.

        This is precisely the verifier family ``V(A_hat, c_t, s_{t+k})`` that
        Proposition 2 proves blind to exogenous change.  It exists here so
        :mod:`sentry.baselines.cached_context` can build the straw man without
        reaching into private state -- not because SENTRY ever uses it.
        """
        return replace(self, state=state, t=t)

    def describe(self) -> str:  # pragma: no cover - cosmetic
        return f"Observation(t={self.t}, images={tuple(self.images.shape)})"


# --------------------------------------------------------------------------
# Action channels
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSpec:
    """Semantics of the ``d_a`` action channels, plus normalisation stats.

    SS2.4.3: "Actions carry channels with different semantics, and we treat
    them differently."  ``C_pos`` and ``C_rot`` index the continuous
    end-effector translation and rotation channels; ``C_grip`` the gripper.

    SS2.9 convention (ii): **all distances in eq. 10 are computed in the
    policy's normalised action space**, using the policy's own statistics, so
    that per-channel thresholds are commensurable.  Every tensor that crosses
    into :mod:`sentry.core.acceptance` is assumed normalised; this class owns
    the conversion so the assumption is checkable in one place.
    """

    d_a: int
    pos: tuple[int, ...]
    rot: tuple[int, ...]
    grip: int
    mean: Tensor
    """``(d_a,)`` per-channel mean of the policy's action statistics."""
    scale: Tensor
    """``(d_a,)`` per-channel scale (std).  Strictly positive."""
    grip_raw_modes: tuple[float, float] = (-1.0, 1.0)
    """Raw encoding of the two gripper modes.  LIBERO uses +-1 (SS2.9 iii)."""

    # -- validation -------------------------------------------------------

    def __post_init__(self) -> None:
        idx = list(self.pos) + list(self.rot) + [self.grip]
        if len(set(idx)) != len(idx):
            raise ValueError(f"pos/rot/grip channel indices overlap: {idx}")
        for i in idx:
            if not (0 <= i < self.d_a):
                raise ValueError(f"channel index {i} outside [0, {self.d_a})")
        if self.mean.shape != (self.d_a,):
            raise ValueError(f"mean must be ({self.d_a},), got {tuple(self.mean.shape)}")
        if self.scale.shape != (self.d_a,):
            raise ValueError(f"scale must be ({self.d_a},), got {tuple(self.scale.shape)}")
        if bool((self.scale <= 0).any()):
            raise ValueError("scale must be strictly positive on every channel")
        self._assert_gripper_sign_test_well_posed()

    def _assert_gripper_sign_test_well_posed(self) -> None:
        """SS2.9 convention (iii).

        LIBERO encodes the two gripper modes as +-1 and "they remain separated
        around zero after standardisation, which is what makes the sign test of
        Sec. 2.4.3 well posed".  If standardisation pushed both modes to the
        same side of zero, ``sign(R[h]_grip) == sign(A_hat[h]_grip)`` would be
        vacuously true and the gripper half of the acceptance rule would be
        silently dead.  Check it instead of assuming it.
        """
        lo, hi = self.grip_raw_modes
        m = float(self.mean[self.grip])
        s = float(self.scale[self.grip])
        z_lo, z_hi = (lo - m) / s, (hi - m) / s
        if z_lo == 0.0 or z_hi == 0.0 or (z_lo > 0) == (z_hi > 0):
            raise ValueError(
                "Gripper sign test is not well posed: raw modes "
                f"{self.grip_raw_modes} standardise to ({z_lo:.4f}, {z_hi:.4f}), "
                "which are not separated around zero.  See Section 2.9, "
                "convention (iii)."
            )

    # -- conversion -------------------------------------------------------

    def normalise(self, raw: Tensor) -> Tensor:
        """Raw action space -> normalised action space."""
        return (raw - self.mean) / self.scale

    def denormalise(self, norm: Tensor) -> Tensor:
        """Normalised action space -> raw action space."""
        return norm * self.scale + self.mean

    def zero_delta_action(self, grip_norm: Tensor) -> Tensor:
        """The ``zero-delta`` pad: null incremental motion, gripper held.

        SS2.4.1: "*zero-delta*, a null incremental motion with the gripper
        channel held".  A null motion is zero in *raw* delta space, which is
        generally **not** zero after normalisation -- hence the round trip.
        The gripper is passed in already normalised because it is held at the
        last surviving action's value, not zeroed.
        """
        raw = torch.zeros(self.d_a, dtype=self.mean.dtype, device=self.mean.device)
        # The statistics may live on the CPU while the chunk is on an
        # accelerator; follow the caller's tensor rather than our own.
        out = self.normalise(raw).clone().to(grip_norm.device)
        out[self.grip] = grip_norm
        return out

    # -- convenience ------------------------------------------------------

    @property
    def real_channels(self) -> tuple[int, ...]:
        """The channels that carry information, in order.

        ``d_a`` is padded -- pi_0 uses 32 while LIBERO fills 7 -- and the padding
        is identically zero, so *any* mean taken over all ``d_a`` channels is
        mostly an average over agreement that costs nothing to produce.

        Measured on ``pi0_libero``: eq. 13's ``L_beh`` reads ``0.0119`` averaged
        over 32 channels and ``0.0531`` over the 7 real ones -- a dilution of
        ``4.46x``, against the ``32/7 = 4.57`` you would get if the padding
        contributed exactly nothing.  A loss reported on the diluted quantity
        looks four and a half times healthier than it is, which is how Stage A
        came to report a 96% reduction while the policy it produced could not do
        the task.
        """
        return tuple(sorted({*self.pos, *self.rot, self.grip}))

    @property
    def pos_idx(self) -> Tensor:
        return torch.as_tensor(self.pos, dtype=torch.long, device=self.mean.device)

    @property
    def rot_idx(self) -> Tensor:
        return torch.as_tensor(self.rot, dtype=torch.long, device=self.mean.device)

    def to(self, device: torch.device | str) -> "ChannelSpec":
        return replace(self, mean=self.mean.to(device), scale=self.scale.to(device))

    @staticmethod
    def libero(d_a: int = 32) -> "ChannelSpec":
        """LIBERO-shaped spec: 3 translation, 3 rotation, 1 gripper, padded.

        Identity normalisation stats by default -- real deployments must pass
        the policy's own statistics (SS2.9 ii).  The +-1 gripper encoding of
        SS2.9 (iii) is preserved.
        """
        return ChannelSpec(
            d_a=d_a,
            pos=(0, 1, 2),
            rot=(3, 4, 5),
            grip=6,
            mean=torch.zeros(d_a),
            scale=torch.ones(d_a),
        )


# --------------------------------------------------------------------------
# Thresholds and check results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """``delta_pos``, ``delta_rot`` -- "the system's only safety-relevant knobs".

    SS2.4.4 sets these by conformal calibration (eq. 12), never by hand.  They
    are passed explicitly through every call rather than read from a module
    global, because paper defect **D3** (eq. 15's ``L_marg`` needs ``d_h``,
    which needs ``delta``, which comes from calibration *after* training) is
    resolved by swapping a provisional ``delta`` in during Stage B.  That only
    works if no module closes over a global.
    """

    pos: float
    rot: float

    def __post_init__(self) -> None:
        if not (self.pos > 0 and self.rot > 0):
            raise ValueError(f"thresholds must be positive, got {self}")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"Thresholds(pos={self.pos:.6g}, rot={self.rot:.6g})"


@dataclass(frozen=True)
class CheckResult:
    """Output of one invocation of the ``Check`` operator (SS2.4).

    ``N`` is the only thing the shallow mode is allowed to produce: "This mode
    never produces an action that is executed.  Its only output is a
    non-negative integer: how many further actions of the current plan may be
    executed." (SS2.3)
    """

    N: int
    """Accepted prefix length, eq. 11.  Always ``0 <= N <= H_k``."""

    margin: Optional[float]
    """``mu = 1 - max_tau d^tau_{N-1}`` when ``N >= 1``, else ``None``.

    Undefined at ``N = 0`` -- SS2.4.3 defines it only "when ``N >= 1``".  We
    return ``None`` so the cascade must handle it explicitly rather than
    silently defaulting to ``0.0`` and comparing against ``mu_esc``.
    """

    d: Tensor
    """``(K, H_k)`` normalised distances ``d^tau_h`` (eq. 10).  Kept for the
    D1 diagnostic and for harvesting calibration statistics."""

    sign_ok: Tensor
    """``(K, H_k)`` bool: gripper sign agreement at each position."""

    H_k: int
    """``H - k``, the number of surviving actions."""

    rung: DepthRung
    """The depth at which this verdict was produced."""

    @property
    def accepted(self) -> bool:
        return self.N > 0

    @property
    def exhausted(self) -> bool:
        """Whether the accepted prefix consumed the entire live suffix."""
        return self.N >= self.H_k

    def is_fragile(self, mu_esc: float) -> bool:
        """Whether SS2.6 would call this verdict fragile and escalate depth.

        "a check that returns ``N = 0`` or ``mu < mu_esc`` is repeated at the
        next depth."
        """
        if self.N == 0:
            return True
        assert self.margin is not None
        return self.margin < mu_esc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mu = "None" if self.margin is None else f"{self.margin:+.4f}"
        return f"CheckResult(N={self.N}/{self.H_k}, mu={mu}, rung={self.rung})"
