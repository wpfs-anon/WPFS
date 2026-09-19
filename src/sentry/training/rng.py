"""Seeded sampling that survives being moved to an accelerator.

``torch.randn(shape, generator=g, device=d)`` requires ``g`` and ``d`` to match:
a CPU generator with a CUDA device raises

    RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

Training code seeds with a plain ``torch.Generator()`` for reproducibility, and
that generator is a CPU one.  Rather than force every caller to build a
device-matched generator -- which would also make a seeded run irreproducible
across devices -- these helpers sample on the generator's own device and then
transfer.  The draw therefore depends on the seed alone, not on where the model
happens to live, which is what
:func:`sentry.training.preflight.check_resume_equivalence` relies on.
"""

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
    """``randn`` of ``shape``, matching ``ref``'s dtype and device."""
    if generator is not None and generator.device.type != ref.device.type:
        out = torch.randn(tuple(shape), generator=generator, dtype=ref.dtype)
        return out.to(ref.device)
    return torch.randn(tuple(shape), generator=generator, dtype=ref.dtype, device=ref.device)


def rand_on(
    shape: Sequence[int] | torch.Size,
    ref: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """``rand`` of ``shape``, matching ``ref``'s dtype and device."""
    if generator is not None and generator.device.type != ref.device.type:
        out = torch.rand(tuple(shape), generator=generator, dtype=ref.dtype)
        return out.to(ref.device)
    return torch.rand(tuple(shape), generator=generator, dtype=ref.dtype, device=ref.device)
