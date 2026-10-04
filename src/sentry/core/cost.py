from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from sentry.core.types import DepthRung

__all__ = ["StageLatencies", "RTX4090D", "l_check", "l_step", "speedup_vs_fixed", "table1"]


@dataclass(frozen=True)
class StageLatencies:

    L_enc: float
    L_pre: float
    L_den: float
    M: int
    L_V: int = 27
    L_B: int = 18

    @property
    def L_full(self) -> float:
        return self.L_enc + self.L_pre + self.M * self.L_den

    @property
    def denoising_bound(self) -> float:
        return self.L_full / (self.L_enc + self.L_pre)


RTX4090D = StageLatencies(L_enc=11.3, L_pre=26.7, L_den=2.0, M=10)


def l_check(rung: DepthRung, lat: StageLatencies) -> float:
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
    if N_exec < 1:
        raise ValueError(f"N_exec must be >= 1, got {N_exec}")
    return (lat.L_full / N_exec) / L_step_bar


_TABLE1_PAPER: tuple[tuple[str, DepthRung, float, float], ...] = (
    ("Full depth (plan mode)", DepthRung(27, 18), 58.0, 1.0),
    ("Cascade rung 1", DepthRung(8, 5), 12.2, 4.8),
    ("Cascade rung 2", DepthRung(8, 9), 19.0, 3.1),
    ("Cascade rung 3", DepthRung(14, 12), 25.7, 2.3),
)


def table1(lat: StageLatencies = RTX4090D) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, rung, paper_ms, paper_speedup in _TABLE1_PAPER:
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


def format_table1(rows: Optional[Sequence[dict[str, object]]] = None) -> str:
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
