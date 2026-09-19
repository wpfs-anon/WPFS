"""LIBERO channel semantics and normalisation (SS2.9, conventions ii and iii).

Spec only -- this module deliberately has no LIBERO dependency, so it can be
imported and checked on a machine with neither the simulator nor a checkpoint.
It records the two facts about LIBERO that the acceptance rule depends on, and
gives the one function you must call with real data before deploying.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from sentry.core.types import ChannelSpec

__all__ = [
    "LIBERO_D_A",
    "libero_spec",
    "fit_from_actions",
    "from_openpi_norm_stats",
    "find_norm_stats",
]

LIBERO_D_A = 32
"""Padded action dimension used by pi_0 (SS2.1: ``d_a = 32``)."""

_POS = (0, 1, 2)
"""End-effector translation delta."""
_ROT = (3, 4, 5)
"""End-effector rotation delta (axis-angle)."""
_GRIP = 6
"""Gripper.  LIBERO encodes the two modes as +-1 (SS2.9, convention iii)."""


def fit_from_actions(actions: Tensor, d_a: int = LIBERO_D_A) -> ChannelSpec:
    """Build a :class:`ChannelSpec` from the policy's own action statistics.

    SS2.9 convention (ii): "all distances in equation 10 are computed in
    normalised space using the policy's own statistics, so that per-channel
    thresholds are commensurable."  This is not a nicety.  A translation delta
    and a rotation delta live on different scales; calibrating ``delta_pos``
    and ``delta_rot`` against unnormalised distances makes the two thresholds
    incomparable and lets one channel group dominate equation 10's ``max``.

    Args:
        actions: ``(N, d_a)`` or ``(N, H, d_a)`` actions in **raw** units --
            ideally the same statistics the policy was trained with, not
            statistics refitted on your evaluation set.

    Constant channels (the padding beyond index 6) get a unit scale rather than
    a degenerate one.

    The returned spec validates that the ``+-1`` gripper modes remain separated
    around zero after standardisation, "which is what makes the sign test of
    Sec. 2.4.3 well posed".  If your checkpoint standardises the gripper onto
    one side of zero, that assertion will fire -- and it should, because the
    gripper half of the acceptance rule would otherwise be silently dead.
    """
    flat = actions.reshape(-1, actions.shape[-1]).float()
    if flat.shape[-1] != d_a:
        raise ValueError(f"expected {d_a} channels, got {flat.shape[-1]}")

    mean = flat.mean(dim=0)
    scale = flat.std(dim=0)
    constant = scale < 1e-6
    mean = torch.where(constant, torch.zeros_like(mean), mean)
    scale = torch.where(constant, torch.ones_like(scale), scale)

    return ChannelSpec(
        d_a=d_a, pos=_POS, rot=_ROT, grip=_GRIP,
        mean=mean, scale=scale, grip_raw_modes=(-1.0, 1.0),
    )


_DEGENERATE = 1e-6
"""Below this, a channel carries no variation and its scale is set to 1.

Not an arbitrary choice: openpi's own ``Unnormalize._unnormalize`` pads missing
scale entries with ``value=1.0``, so unit scale is what the policy itself
applies to channels its statistics do not cover.
"""


def find_norm_stats(checkpoint_dir: str | Path) -> Path:
    """Locate ``norm_stats.json`` inside a downloaded openpi checkpoint.

    The file sits under ``assets/<repo>/<dataset>/norm_stats.json`` and the
    middle components vary by checkpoint, so we search rather than hardcode.
    """
    root = Path(checkpoint_dir)
    hits = sorted(root.glob("**/norm_stats.json"))
    if not hits:
        raise FileNotFoundError(f"no norm_stats.json under {root}")
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} norm_stats.json files under {root}; pass one explicitly: "
            + ", ".join(str(h) for h in hits)
        )
    return hits[0]


def from_openpi_norm_stats(
    path: str | Path,
    d_a: int = LIBERO_D_A,
    use_quantiles: bool = False,
) -> ChannelSpec:
    """Build a :class:`ChannelSpec` from a real openpi ``norm_stats.json``.

    This is the function SS3.9 convention (ii) actually calls for -- "the
    policy's own statistics" -- and it exists because the obvious alternatives
    both fail on a real checkpoint:

    - :func:`fit_from_actions` refits statistics on *your* data, which is what
      its own docstring warns against;
    - handing the raw file to :class:`ChannelSpec` raises, because ``pi0_libero``
      stores ``actions/std = 0.0`` for channels 7..31 (the padding that brings
      LIBERO's 7 real channels up to ``d_a = 32``) and ``ChannelSpec`` requires a
      strictly positive scale.

    ``use_quantiles`` must match the policy. openpi selects it by model type --
    ``use_quantile_norm = model_type != PI0`` -- so **pi0 uses z-score** (the
    default here) while pi0.5 and pi0-FAST use quantiles.  Getting this wrong is
    silent: both produce plausible numbers on different scales, and every
    threshold eq. 12 calibrates would be expressed in the wrong units.

    The quantile branch is written as an equivalent affine map so that both
    conventions reduce to ``ChannelSpec``'s single ``(x - mean) / scale``:
    openpi computes ``(x - q01)/(q99 - q01) * 2 - 1``, which is exactly
    ``(x - (q01+q99)/2) / ((q99-q01)/2)``.
    """
    stats = json.loads(Path(path).read_text())
    # openpi nests one level under "norm_stats"; tolerate a flat file too.
    node = stats.get("norm_stats", stats)
    if "actions" not in node:
        raise ValueError(f"{path} has no 'actions' entry; keys: {sorted(node)}")
    act = node["actions"]

    def vec(key: str) -> Tensor:
        if key not in act:
            raise ValueError(
                f"{path} lacks 'actions/{key}', required for "
                f"use_quantiles={use_quantiles}"
            )
        t = torch.as_tensor(act[key], dtype=torch.float32)
        # pi0_libero's file carries all d_a channels, zero beyond LIBERO's 7;
        # pi05_libero's carries only the 7 real ones.  Zero-padding the short
        # form sends the padding through the degenerate branch below, exactly
        # as pi0's zeros already go.
        if t.ndim == 1 and t.shape[0] < d_a:
            t = torch.cat([t, torch.zeros(d_a - t.shape[0])])
        if t.shape != (d_a,):
            raise ValueError(f"actions/{key} has shape {tuple(t.shape)}, expected ({d_a},)")
        return t

    if use_quantiles:
        q01, q99 = vec("q01"), vec("q99")
        mean = (q01 + q99) / 2.0
        scale = (q99 - q01) / 2.0
    else:
        mean, scale = vec("mean"), vec("std")

    # Channels the policy's statistics do not actually cover -- the d_a padding.
    # Unit scale (not openpi's ``std + 1e-6``) is deliberate: at 1e-6 the float
    # noise a real model emits into an unused channel is amplified a millionfold,
    # and Definition 1's liveness norm is taken over the *full* action vector, so
    # that noise would swamp the very quantity the label depends on.
    degenerate = scale.abs() < _DEGENERATE
    mean = torch.where(degenerate, torch.zeros_like(mean), mean)
    scale = torch.where(degenerate, torch.ones_like(scale), scale)

    return ChannelSpec(
        d_a=d_a, pos=_POS, rot=_ROT, grip=_GRIP,
        mean=mean, scale=scale, grip_raw_modes=(-1.0, 1.0),
    )


def libero_spec(
    mean: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    d_a: int = LIBERO_D_A,
) -> ChannelSpec:
    """A LIBERO-shaped spec, defaulting to identity normalisation.

    The identity default is for tests and shape-checking **only**.  Real
    deployments must pass the policy's own statistics -- use
    :func:`fit_from_actions`.
    """
    return ChannelSpec(
        d_a=d_a, pos=_POS, rot=_ROT, grip=_GRIP,
        mean=torch.zeros(d_a) if mean is None else mean,
        scale=torch.ones(d_a) if scale is None else scale,
        grip_raw_modes=(-1.0, 1.0),
    )
