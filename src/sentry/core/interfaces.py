"""The single seam between SENTRY's algorithm and a concrete VLA.

Nothing in :mod:`sentry.core` may import a concrete model.  Everything the
check operator, the cascade and the execution loop need from a policy is
declared here.
"""

from __future__ import annotations

import inspect
from typing import Optional, Protocol, runtime_checkable

from torch import Tensor

from sentry.core.types import Observation

__all__ = ["VLABackend", "Environment", "assert_no_cache_seam"]


@runtime_checkable
class VLABackend(Protocol):
    """A pi_0-style flow-matching VLA exposed at two depths (SS2.3).

    Implementations must satisfy **Proposition 4**: with ``adapters=False`` the
    network is numerically identical to the pretrained policy, because theta is
    never updated and Delta contributes zero when gated off.  This is tested by
    exact tensor equality, not ``allclose``.
    """

    # -- geometry, read off the checkpoint --------------------------------

    L_V: int
    """Vision encoder depth.  SigLIP-So400m: 27."""

    L_B: int
    """Backbone depth.  Gemma-2B: 18.

    SS2.9 convention (iv): ``L_V`` and ``L_B`` "must be read off the
    checkpoint, since VLM backbones in this family vary between 18 and 32
    layers."  Never hardcode them in :mod:`sentry.core`.
    """

    H: int
    """Chunk length."""

    d_a: int
    """Padded action dimension."""

    M: int
    """Euler steps used by plan mode."""

    # -- the two modes ----------------------------------------------------

    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        """Plan mode ``Pi_deep`` (SS2.3).

        All ``L_V`` encoder layers, all ``L_B`` backbone layers, adapters
        **disabled**, ``M`` solver steps.  Integrates the learned velocity
        field from Gaussian noise (eq. 2) to tau = 1.

        Args:
            obs: the conditioning observation.
            noise: optional ``(H, d_a)`` draw to use as ``A^0`` instead of a
                fresh one.  Omit it for deployment; supply it when two plans
                must be **comparable**.

        Returns ``(H, d_a)`` in the policy's **normalised** action space.

        **Paper defect D5 -- Definition 1 is ill-posed without this argument.**
        Eq. 5 writes ``|| [pi_theta(o_{t+k})]_h - A_tilde[h] || <= epsilon`` as
        though ``pi_theta(o)`` were a value.  For a flow-matching policy it is a
        *sample* from a multi-modal distribution -- which is the entire reason
        the target is a diffusion policy rather than a regressor.  Two calls on
        the **same** observation therefore disagree by the policy's own sampling
        spread, and on an untrained ``TinyPi0`` that spread measures ~4.0 per
        position against ``liveness_eps = 0.05``.  Comparing independent draws
        makes eq. 5 a measurement of sampling variance, not of staleness, and
        the liveness label is the foundation of *both* eq. 12's calibration and
        eq. 15's ``h_star``.

        The well-posed reading is a **common random number**: draw ``epsilon``
        once for the chunk under test and re-use it for every fresh plan the
        label is computed against, so the only thing that differs between the
        two plans is the observation.  That is what this argument exists for;
        :func:`sentry.core.calibration.liveness` documents the caller's side of
        the contract.
        """
        ...

    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        """A single velocity evaluation ``v(A^tau, tau | o)``.

        Args:
            A_tau: ``(K, H, d_a)`` interpolants, batched over ``T``.
            tau: ``(K,)`` the model's own time argument (already mapped through
                the tau convention by :mod:`sentry.core.renoise` -- backends
                receive whatever direction they natively integrate in).
            obs: conditioning observation.  For a check this is the **fresh**
                ``o_{t+k}``; recomputing it from the current images is what
                Corollary 3 requires.
            E_V: encoder layer-prefix budget, ``1 <= E_V <= L_V``.
            E_B: backbone layer-prefix budget, ``1 <= E_B <= L_B``.  A single
                budget governs perception *and* denoising (SS2.9).
            adapters: whether the LoRA adapters ``Delta`` are gated on.

        Returns ``(K, H, d_a)``.

        Truncation is a **runtime prefix**, not a rebuilt model: plan mode and
        check mode alternate within a single episode, so ``E_V``/``E_B`` and
        ``adapters`` are per-call arguments.  For the same reason SS2.9 keeps
        adapters "unmerged and gated by a boolean, since plan mode and check
        mode alternate within a single episode; merging would require a weight
        copy per switch."

        **This signature deliberately admits no KV cache.**  SS2.9: "Plan and
        check differ in depth, in adapter state, and in input image.  The
        prefix KV computed in plan mode is therefore not a valid cache for
        check mode, and reusing it is a correctness error, not an
        optimisation."  Leaving the cache out of the signature makes that
        structural rather than a comment.  (Within a single check the ``K``
        interpolants share the prefix and are batched -- that is internal to
        the implementation.)
        """
        ...


@runtime_checkable
class Environment(Protocol):
    """Minimal environment contract for Algorithm 1."""

    def observe(self) -> Observation:
        """Return the observation **as of now**.

        The execution loop calls this inside the speculative phase, once per
        check.  Freshness is asserted via ``Observation.t``.
        """
        ...

    def step(self, action: Tensor) -> None:
        """Execute one action (normalised action space)."""
        ...

    @property
    def terminated(self) -> bool: ...

    @property
    def t(self) -> int:
        """Current environment timestep."""
        ...


def assert_no_cache_seam(backend: object) -> None:
    """Fail loudly if a backend tries to smuggle a KV cache through the seam.

    SS2.9 calls cache reuse across modes "a correctness error, not an
    optimisation".  The protocol prevents it by omission; this check catches
    an implementation that widened the signature anyway.
    """
    sig = inspect.signature(backend.velocity)  # type: ignore[attr-defined]
    banned = {"cache", "kv", "kv_cache", "past_key_values", "context", "ctx"}
    offending = sorted(banned.intersection(sig.parameters))
    if offending:
        raise TypeError(
            f"{type(backend).__name__}.velocity exposes {offending}: plan mode "
            "and check mode differ in depth, adapter state and input image, so "
            "no cache may be shared between them (Section 2.9)."
        )
