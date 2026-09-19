r"""The cost model -- equations 17 and 18 (SS2.7).

.. math::
    L_{check}(E_V, E_B) &\approx \frac{E_V}{L_V} L_{enc}
                          + \frac{E_B}{L_B} L_{pre} + L_{den}                \\
    \bar{L}_{step} &= \frac{\rho(L_{full} + \bar{J}\bar{L}_{check})
                             + (1-\rho)\bar{J}\bar{L}_{check}}
                            {\rho\, m_{min} + (1-\rho)\bar{N}}

Equation 17's last term is a **single** batched velocity evaluation covering
all ``K`` verification timesteps, not ``K`` of them (SS2.4.2).  Adapter overhead
is ``O(r/d)`` per projection and is neglected.

The section's two accounting points are the reason this module exists as more
than a formula:

1. "a shallow check is *not* cheaper per call than a small dedicated drafter: a
   110M drafter is roughly 25x smaller than the backbone, whereas prefix
   truncation buys a factor of at most ``L_B/E_B``.  SENTRY is therefore not a
   claim about ``L_check`` but about equation 18... and the product
   ``rho^-1 N_bar`` is where the gain must appear.  **This is stated as a
   prediction so that it can fail.**"
2. "prior work runs the *full* vision encoder on the flash path in order to
   feed its drafter, and skips only the backbone prefill; SENTRY truncates the
   encoder as well, and spends the savings on a shallow but fresh prefill.  The
   exchange is close to cost-neutral and buys sensitivity to exogenous change."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from sentry.core.types import DepthRung

__all__ = ["StageLatencies", "RTX4090D", "l_check", "l_step", "speedup_vs_fixed", "table1"]


@dataclass(frozen=True)
class StageLatencies:
    """Per-stage latencies of one replanning round (SS2.1, eq. 4).

    ``L_full = L_enc + L_pre + M * L_den``.  The first two terms are
    compute-bound and dominate; the third is memory-bound and leaves the
    accelerator underutilised.
    """

    L_enc: float
    """Vision encoder."""
    L_pre: float
    """One backbone prefill over O(10^3) multimodal tokens."""
    L_den: float
    """One action-expert step."""
    M: int
    """Solver steps used by plan mode."""
    L_V: int = 27
    L_B: int = 18

    @property
    def L_full(self) -> float:
        """Equation 4."""
        return self.L_enc + self.L_pre + self.M * self.L_den

    @property
    def denoising_bound(self) -> float:
        """Ceiling on any acceleration that touches only ``M`` (SS2.1).

        "any acceleration that touches only ``M`` is bounded by
        ``(L_enc + L_pre)^-1 L_full`` regardless of how aggressively the
        sampler is distilled."
        """
        return self.L_full / (self.L_enc + self.L_pre)


RTX4090D = StageLatencies(L_enc=11.3, L_pre=26.7, L_den=2.0, M=10)
"""Table 1's profile: ``L_enc = 11.3 ms``, ``L_pre = 26.7 ms``,
``M*L_den = 20.0 ms``, hence ``L_den = 2.0 ms`` and ``L_full = 58.0 ms``."""


def l_check(rung: DepthRung, lat: StageLatencies) -> float:
    """Equation 17 -- cost of one shallow check at a given depth."""
    return (
        (rung.E_V / lat.L_V) * lat.L_enc
        + (rung.E_B / lat.L_B) * lat.L_pre
        + lat.L_den
    )


def l_step(
    rho: float,
    N_bar: float,
    J_bar: float,
    m_min: int,
    lat: StageLatencies,
    L_check_bar: float,
) -> float:
    """Equation 18 -- amortised cost per environment step.

    Args:
        rho: fraction of checks that end in rejection.
        N_bar: mean accepted prefix.
        J_bar: mean number of cascade rungs evaluated per check, ``>= 1``.
        m_min: unconditional commit length.
        lat: stage latencies (supplies ``L_full``).
        L_check_bar: mean cost of a single rung's check.

    This is a renewal-reward ratio: expected cost per round over expected
    environment steps per round.  A rejecting round pays a full replan *plus*
    the checks that led to the rejection, and advances by ``m_min``; an
    accepting round pays only the checks and advances by ``N_bar``.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must lie in [0,1], got {rho}")
    if J_bar < 1.0:
        raise ValueError(f"J_bar must be >= 1, got {J_bar}")

    numerator = rho * (lat.L_full + J_bar * L_check_bar) + (1.0 - rho) * J_bar * L_check_bar
    denominator = rho * m_min + (1.0 - rho) * N_bar
    if denominator <= 0.0:
        raise ValueError(
            f"degenerate step rate: rho={rho}, m_min={m_min}, N_bar={N_bar}"
        )
    return numerator / denominator


def speedup_vs_fixed(L_step_bar: float, N_exec: int, lat: StageLatencies) -> float:
    """Speedup against a baseline that replans every ``N_exec`` steps.

    "The baseline replans every ``N_exec`` steps at cost ``L_tgt``, giving
    ``L_tgt / N_exec`` per step, and ``S`` is the ratio." (Appendix A.2)
    """
    if N_exec < 1:
        raise ValueError(f"N_exec must be >= 1, got {N_exec}")
    return (lat.L_full / N_exec) / L_step_bar


# --------------------------------------------------------------------------
# Table 1 reproduction
# --------------------------------------------------------------------------

_TABLE1_PAPER: tuple[tuple[str, DepthRung, float, float], ...] = (
    ("Full depth (plan mode)", DepthRung(27, 18), 58.0, 1.0),
    ("Cascade rung 1", DepthRung(8, 5), 12.2, 4.8),
    ("Cascade rung 2", DepthRung(8, 9), 19.0, 3.1),
    ("Cascade rung 3", DepthRung(14, 12), 25.7, 2.3),
)


def table1(lat: StageLatencies = RTX4090D) -> list[dict[str, object]]:
    """Recompute Table 1 from equation 17 and compare against the paper.

    Table 1's caption is explicit that "All entries are projections from
    equation 17, not measurements", so this is a pure arithmetic check.

    Rungs 1 and 2 do **not** reproduce: eq. 17 gives 12.77 ms and 18.70 ms
    where the table prints 12.2 and 19.0.  Rung 3 reproduces exactly (25.66 ->
    25.7), as does the full-depth row.  The discrepancy is small and does not
    affect any claim -- the paper's own 58.0/12.2 = 4.8x is internally
    consistent -- but it is surfaced rather than hidden, since anyone
    reimplementing eq. 17 will hit it.
    """
    rows: list[dict[str, object]] = []
    for name, rung, paper_ms, paper_speedup in _TABLE1_PAPER:
        # The full-depth row is a plan-mode round (M solver steps), not a check.
        computed = lat.L_full if rung.E_B == lat.L_B and rung.E_V == lat.L_V else l_check(rung, lat)
        rows.append(
            {
                "configuration": name,
                "E_V/L_V": f"{rung.E_V}/{lat.L_V}",
                "E_B/L_B": f"{rung.E_B}/{lat.L_B}",
                "paper_ms": paper_ms,
                "computed_ms": round(computed, 2),
                "delta_ms": round(computed - paper_ms, 2),
                "paper_speedup": paper_speedup,
                "computed_speedup": round(lat.L_full / computed, 2),
            }
        )
    return rows


def format_table1(rows: Optional[Sequence[dict[str, object]]] = None) -> str:  # pragma: no cover
    """Render :func:`table1` for the terminal."""
    rows = rows if rows is not None else table1()
    head = f"{'Configuration':<24} {'E_V/L_V':>8} {'E_B/L_B':>8} {'paper':>8} {'eq.17':>8} {'delta':>7} {'speedup':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['configuration']:<24} {r['E_V/L_V']:>8} {r['E_B/L_B']:>8} "
            f"{r['paper_ms']:>8.1f} {r['computed_ms']:>8.2f} {r['delta_ms']:>+7.2f} "
            f"{r['computed_speedup']:>8.2f}x"
        )
    return "\n".join(lines)
