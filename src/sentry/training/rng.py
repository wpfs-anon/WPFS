from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = ["randn_like_ref", "rand_on"]


def randn_like_ref(
    shape: Sequence[int] | torch.Size,
    ref: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if generator is not None and generator.device.type != ref.device.type:
        out = torch.randn(tuple(shape), generator=generator, dtype=ref.dtype)
        return out.to(ref.device)
    return torch.randn(tuple(shape), generator=generator, dtype=ref.dtype, device=ref.device)


def rand_on(
    shape: Sequence[int] | torch.Size,
    ref: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if generator is not None and generator.device.type != ref.device.type:
        out = torch.rand(tuple(shape), generator=generator, dtype=ref.dtype)
        return out.to(ref.device)
    return torch.rand(tuple(shape), generator=generator, dtype=ref.dtype, device=ref.device)
