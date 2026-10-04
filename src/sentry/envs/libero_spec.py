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

_POS = (0, 1, 2)
_ROT = (3, 4, 5)
_GRIP = 6


def fit_from_actions(actions: Tensor, d_a: int = LIBERO_D_A) -> ChannelSpec:
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


def find_norm_stats(checkpoint_dir: str | Path) -> Path:
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
    stats = json.loads(Path(path).read_text())
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
    return ChannelSpec(
        d_a=d_a, pos=_POS, rot=_ROT, grip=_GRIP,
        mean=torch.zeros(d_a) if mean is None else mean,
        scale=torch.ones(d_a) if scale is None else scale,
        grip_raw_modes=(-1.0, 1.0),
    )
