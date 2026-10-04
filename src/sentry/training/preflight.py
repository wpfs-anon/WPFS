from __future__ import annotations

import copy
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from sentry.config import SentryConfig
from sentry.core import acceptance
from sentry.core.renoise import interpolate, reconstruct, to_model_time
from sentry.core.types import Observation, Thresholds
from sentry.envs.mock_env import MockReachConfig, fit_spec
from sentry.eval.harness import SampleGenConfig
from sentry.models.lora import LoRALinear, adapters as adapters_ctx, set_adapters
from sentry.models.toy_pi0 import TinyPi0, TinyPi0Config
from sentry.training import checkpoint
from sentry.training.datagen import make_stage_a_batch, make_stage_b_batch
from sentry.training.losses import margin_loss, sensitivity_loss
from sentry.training.params import audit, split_adapters
from sentry.training.stage_a import StageAConfig, StageATrainer
from sentry.training.stage_b import StageBConfig, StageBTrainer

__all__ = ["Result", "run_all", "main"]


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0
    advisory: bool = False


def _rig(H: int = 8):
    torch.manual_seed(0)
    env_cfg = MockReachConfig(H=H, img_size=16, max_steps=60)
    spec = fit_spec(env_cfg, n=64)
    model = TinyPi0(
        TinyPi0Config(H=H, d_a=7, img_size=16, L_V=6, L_B=6,
                      lora_layers_V=4, lora_layers_B=5)
    )
    cfg = SentryConfig(
        H=H, m_min=2, taus=(0.6, 0.9), p_depth=(2, 5),
        ladder=SentryConfig().ladder[:1],
    )
    return model, cfg, spec, env_cfg


def _stage_a(model, cfg, steps=50):
    return StageATrainer(model, cfg, StageAConfig(E_V=3, total_steps=steps, lr=3e-3))


def _stage_b(model, cfg, spec, steps=50):
    return StageBTrainer(model, cfg, StageBConfig(E_V=3, total_steps=steps, lr=3e-3), spec)


def check_adapter_partition() -> Result:
    model, *_ = _rig()
    g = split_adapters(model)
    ids_V = {id(p) for p in g.params_V()}
    ids_B = {id(p) for p in g.params_B()}
    overlap = ids_V & ids_B
    total = sum(1 for n, _ in model.named_parameters() if ".lora_" in n)
    ok = not overlap and len(ids_V) + len(ids_B) == total
    return Result(
        "adapter partition (Delta_V / Delta_B disjoint and complete)",
        ok,
        f"Delta_V={len(ids_V)} Delta_B={len(ids_B)} total={total} overlap={len(overlap)}",
    )


def check_stage_a_gradients() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_a(model, cfg)
    batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))
    for _ in range(2):
        tr.step(batch, torch.Generator().manual_seed(0))

    tr.opt.zero_grad(set_to_none=True)
    loss, _ = tr.loss_on(batch, torch.Generator().manual_seed(0))
    loss.backward()
    rep = audit(model, "Stage A", tr.groups.delta_V)
    return Result("Stage A gradient audit (only Delta_V moves)", rep.ok, str(rep))


def check_stage_b_gradients() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    rng = random.Random(0)
    for _ in range(2):
        tr.step(batch, rng)

    tr.opt.zero_grad(set_to_none=True)
    loss, _ = tr.loss_on(batch, rng, torch.Generator().manual_seed(0), E_B=model.L_B)
    loss.backward()
    rep = audit(model, "Stage B", tr.groups.delta_B)
    return Result("Stage B gradient audit (Delta_V frozen)", rep.ok, str(rep))


def check_theta_never_moves() -> Result:
    model, cfg, spec, env_cfg = _rig()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if ".lora_" not in n}

    tr = _stage_a(model, cfg)
    batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))
    for _ in range(3):
        tr.step(batch, torch.Generator().manual_seed(0))

    moved = [n for n, v in before.items() if not torch.equal(v, dict(model.named_parameters())[n])]
    return Result(
        "theta frozen through training (Prop. 4 precondition)",
        not moved,
        "no base weight changed" if not moved else f"MOVED: {moved[:3]}",
    )


def check_prop4_after_training() -> Result:
    model, cfg, spec, env_cfg = _rig()
    o = make_stage_a_batch(model, cfg, spec, 1, env_cfg, random.Random(0)).obs[0]
    A = torch.randn(1, model.H, model.d_a)
    tau = torch.tensor([0.5])

    before = model.velocity(A, tau, o, model.L_V, model.L_B, adapters=False)

    tr = _stage_a(model, cfg)
    batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))
    for _ in range(3):
        tr.step(batch, torch.Generator().manual_seed(0))

    after = model.velocity(A, tau, o, model.L_V, model.L_B, adapters=False)
    ok = torch.equal(before, after)
    return Result(
        "Proposition 4 after training (exact, not allclose)",
        ok,
        "plan mode bit-identical" if ok else f"max drift {float((after-before).abs().max()):.3e}",
    )


def check_teacher_detached() -> Result:
    model, cfg, spec, env_cfg = _rig()
    o = make_stage_a_batch(model, cfg, spec, 1, env_cfg, random.Random(0)).obs[0]
    A_tau = torch.randn(1, model.H, model.d_a)
    tau = torch.tensor([0.5])

    with torch.no_grad(), adapters_ctx(model, False):
        v_full = model.velocity(A_tau, tau, o, model.L_V, model.L_B, adapters=False)
    v_shal = model.velocity(A_tau, tau, o, 3, 3, adapters=True)

    ok = v_full.grad_fn is None and v_shal.grad_fn is not None
    return Result(
        "teacher branch detached (no_grad), student attached",
        ok,
        f"teacher grad_fn={v_full.grad_fn}, student grad_fn={type(v_shal.grad_fn).__name__}",
    )


def check_tau_from_deployed_schedule() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    seen = set()
    real_interp = torch.Tensor.mul

    import sentry.training.stage_b as sb
    original = sb.interpolate

    def spy(A_hat, s, eps):
        seen.update(round(float(x), 6) for x in s)
        return original(A_hat, s, eps)

    sb.interpolate = spy
    try:
        for _ in range(12):
            tr.loss_on(batch, random.Random(1), torch.Generator().manual_seed(0))
    finally:
        sb.interpolate = original

    allowed = {round(float(t), 6) for t in cfg.taus}
    ok = seen and seen.issubset(allowed)
    return Result(
        "Stage B tau drawn from T, not U(0,1)",
        bool(ok),
        f"observed tau values {sorted(seen)} vs T={sorted(allowed)}",
    )


def check_depth_curriculum() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    rng = random.Random(0)
    depths = [tr.sample_depth(rng) for _ in range(400)]
    lo, hi = cfg.p_depth
    support = set(range(lo, min(hi, model.L_B) + 1))
    seen = set(depths)
    return Result(
        "depth curriculum covers p_depth",
        seen == support,
        f"sampled {sorted(seen)} over support {sorted(support)}",
    )


def check_margin_loss_edges() -> Result:
    d_good = torch.full((6,), 0.1)
    fully_live = margin_loss(d_good, h_star=6, H_k=6, m=0.2)

    d_bad = torch.full((6,), 0.1)
    at_zero = margin_loss(d_bad, h_star=0, H_k=6, m=0.2)

    ok = (
        torch.isfinite(fully_live) and float(fully_live) == 0.0
        and torch.isfinite(at_zero) and float(at_zero) > 0.0
    )
    return Result(
        "eq. 15 edge cases (h*=H_k omits push-up; h*=0 no div-by-zero)",
        bool(ok),
        f"h*=H_k -> {float(fully_live):.4f} (want 0), h*=0 -> {float(at_zero):.4f} (want >0)",
    )


def check_anti_collapse_fires() -> Result:
    eta = 0.05
    collapsed = torch.randn(4, 8, 7)
    penalty_collapsed = sensitivity_loss(collapsed, collapsed.clone(), eta)
    penalty_healthy = sensitivity_loss(collapsed, collapsed + 5.0, eta)

    ok = abs(float(penalty_collapsed) - eta) < 1e-6 and float(penalty_healthy) == 0.0
    return Result(
        "eq. 16 penalises an observation-ignoring verifier",
        ok,
        f"collapsed -> {float(penalty_collapsed):.4f} (want {eta}), "
        f"sensitive -> {float(penalty_healthy):.4f} (want 0)",
    )


def check_finite_and_deterministic() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    l1, _ = tr.loss_on(batch, random.Random(7), torch.Generator().manual_seed(3))
    l2, _ = tr.loss_on(batch, random.Random(7), torch.Generator().manual_seed(3))
    ok = torch.isfinite(l1) and torch.isfinite(l2) and torch.allclose(l1, l2)
    return Result(
        "losses finite and reproducible under a fixed seed",
        bool(ok),
        f"loss={float(l1.detach()):.6f} vs {float(l2.detach()):.6f}",
    )


def check_overfit_single_batch() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_a(model, cfg, steps=60)
    batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))

    first = None
    last = None
    for i in range(60):
        parts = tr.step(batch, torch.Generator().manual_seed(0))
        first = parts["loss"] if first is None else first
        last = parts["loss"]

    drop = (first - last) / abs(first) if first else 0.0
    return Result(
        "Stage A overfits a single batch",
        drop > 0.20,
        f"loss {first:.5f} -> {last:.5f}  ({drop:+.1%})",
    )


def check_plan_is_deterministic_under_shared_noise() -> Result:
    model, cfg, spec, env_cfg = _rig()
    batch = make_stage_a_batch(model, cfg, spec, 1, env_cfg, random.Random(0))
    obs = batch.obs[0]

    noise = torch.randn(model.H, model.d_a)
    same = float((model.plan(obs, noise) - model.plan(obs, noise)).abs().max())
    free = float(
        torch.linalg.vector_norm(model.plan(obs) - model.plan(obs), dim=-1).mean()
    )
    return Result(
        "plan(obs, noise) is deterministic (paper defect D5)",
        same == 0.0,
        f"shared noise: max|diff| = {same:.3e}  |  "
        f"independent draws: mean L2 = {free:.3f} vs liveness_eps = {cfg.liveness_eps}",
    )


def check_liveness_labels_are_informative() -> Result:
    model, cfg, spec, env_cfg = _rig()
    batch = make_stage_b_batch(
        model, cfg, spec, 8, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    h = batch.h_star.tolist()
    H_k = batch.H_k.tolist()
    kinds = batch.kinds

    pos_bad = [
        (k, hi, hk) for k, hi, hk in zip(kinds, h, H_k)
        if k == "positive" and hi != hk
    ]
    n4_bad = [hi for k, hi in zip(kinds, h) if k == "N4" and hi < 1]
    degenerate = len(set(h)) <= 1

    ok = not pos_bad and not n4_bad and not degenerate
    detail = f"h*={h}  H_k={H_k}  kinds={list(kinds)}"
    if pos_bad:
        detail += f" | {len(pos_bad)} positive(s) with h* != H_k"
    if n4_bad:
        detail += f" | {len(n4_bad)} N4 sample(s) with h* < 1"
    if degenerate:
        detail += " | every label identical"
    return Result("Stage B: h_star labels are informative", ok, detail)


def check_negative_generators_are_all_present() -> Result:
    model, cfg, spec, env_cfg = _rig()
    batch = make_stage_b_batch(
        model, cfg, spec, 24,
        SampleGenConfig(env=env_cfg, min_H_k=3, j_min=1),
        random.Random(1),
    )
    seen = {k for k in batch.kinds if k != "positive"}
    missing = sorted({"N1", "N2", "N3", "N4"} - seen)
    counts = {k: batch.kinds.count(k) for k in sorted(set(batch.kinds))}
    return Result(
        "Stage B: all four negative generators appear",
        not missing,
        f"{counts}" + (f"  | MISSING {missing}" if missing else ""),
    )


def check_stage_b_distillation_can_converge() -> Result:
    model, cfg, spec, env_cfg = _rig()
    cfg = cfg.with_(lambda_m=0.0, lambda_s=0.0)
    tr = _stage_b(model, cfg, spec, steps=150)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    rng = random.Random(0)

    first = None
    for _ in range(150):
        tr.opt.zero_grad(set_to_none=True)
        loss, parts = tr.loss_on(batch, rng, E_B=4)
        loss.backward()
        tr.opt.step()
        first = parts["L_dist"] if first is None else first
    last = parts["L_dist"]

    drop = (first - last) / abs(first) if first else 0.0
    return Result(
        "Stage B: L_dist converges with competing terms off",
        drop > 0.50,
        f"L_dist {first:.5f} -> {last:.5f}  ({drop:+.1%})",
    )


def check_margin_target_is_reachable() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 6, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    th = Thresholds(pos=tr.b_cfg.delta_provisional, rot=tr.b_cfg.delta_provisional)
    rng = random.Random(0)

    teacher, student = [], []
    for i, obs in enumerate(batch.obs):
        A_hat = batch.A_hat[i]
        H_k, h_star = int(batch.H_k[i]), int(batch.h_star[i])
        tau = torch.tensor([cfg.taus[rng.randrange(len(cfg.taus))]])
        eps = torch.randn(A_hat.shape)
        A_tau = interpolate(A_hat, tau, eps)
        tau_m = to_model_time(tau, cfg.tau_convention)

        with torch.no_grad(), adapters_ctx(model, False):
            v_t = model.velocity(A_tau, tau_m, obs, model.L_V, model.L_B, adapters=False)
            v_s = model.velocity(A_tau, tau_m, obs, tr.b_cfg.E_V, 4, adapters=True)

        for v, sink in ((v_t, teacher), (v_s, student)):
            R = reconstruct(A_tau, tau, v, cfg.tau_convention)
            d = acceptance.normalised_distances(R, A_hat, spec, th, H_k).max(dim=0).values
            sink.append(float(margin_loss(d, h_star, H_k, cfg.margin_m)))

    t_bar = sum(teacher) / len(teacher)
    s_bar = sum(student) / len(student)
    verdict = "compatible" if t_bar < 0.1 else "RIVALS -- teacher fails its own hinge"
    return Result(
        "eq. 15's hinge is reachable by the teacher (advisory)",
        True,
        f"L_marg teacher {t_bar:.4f} vs student {s_bar:.4f}  -> {verdict}",
        advisory=True,
    )


def check_checkpoint_roundtrip() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_a(model, cfg)
    batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))
    for _ in range(4):
        tr.step(batch, torch.Generator().manual_seed(0))

    with tempfile.TemporaryDirectory() as tmp:
        p = checkpoint.save(Path(tmp) / "a.pt", tr, stage="A")
        saved = {n: v.clone() for n, v in tr.state_dict()["adapters"].items()}
        opt_before = len(tr.opt.state_dict()["state"])

        for _ in range(4):
            tr.step(batch, torch.Generator().manual_seed(0))
        checkpoint.load(p, tr, strict_stage="A")

        restored = tr.state_dict()["adapters"]
        same = all(torch.equal(saved[n], restored[n]) for n in saved)
        opt_after = len(tr.opt.state_dict()["state"])
        step_ok = tr.step_idx == 4
        found = checkpoint.latest(tmp, stage="A")

    ok = same and step_ok and opt_after == opt_before and found is not None
    return Result(
        "checkpoint round-trip (adapters + optimiser + step)",
        bool(ok),
        f"adapters restored={same}, step_idx={tr.step_idx} (want 4), "
        f"optimiser slots {opt_before}->{opt_after}",
    )


def check_resume_equivalence() -> Result:
    def run(interrupt: bool):
        model, cfg, spec, env_cfg = _rig()
        tr = _stage_a(model, cfg)
        batch = make_stage_a_batch(model, cfg, spec, 2, env_cfg, random.Random(0))
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(10):
                if interrupt and i == 5:
                    p = checkpoint.save(Path(tmp) / "r.pt", tr, stage="A")
                    checkpoint.load(p, tr, strict_stage="A")
                tr.step(batch, torch.Generator().manual_seed(i))
        return [v.clone() for _, v in tr.groups.delta_V]

    straight, resumed = run(False), run(True)
    diffs = [float((a - b).abs().max()) for a, b in zip(straight, resumed)]
    worst = max(diffs) if diffs else 0.0
    return Result(
        "resume equivalence (10 steps == 5 + checkpoint + 5)",
        worst < 1e-6,
        f"max parameter divergence {worst:.3e}",
    )


def check_throughput() -> Result:
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    rng = random.Random(0)
    tr.step(batch, rng)
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        tr.step(batch, rng)
    per_step = (time.perf_counter() - t0) / n

    hours = (20_000 + 60_000) * per_step / 3600
    return Result(
        "throughput projection (advisory)",
        True,
        f"{per_step*1000:.0f} ms/step on this toy CPU model -> "
        f"{hours:.1f} h for 20k+60k steps; Colab caps a session near 12 h",
        advisory=True,
    )


CHECKS: tuple[Callable[[], Result], ...] = (
    check_adapter_partition,
    check_stage_a_gradients,
    check_stage_b_gradients,
    check_theta_never_moves,
    check_prop4_after_training,
    check_teacher_detached,
    check_tau_from_deployed_schedule,
    check_depth_curriculum,
    check_margin_loss_edges,
    check_anti_collapse_fires,
    check_finite_and_deterministic,
    check_overfit_single_batch,
    check_plan_is_deterministic_under_shared_noise,
    check_liveness_labels_are_informative,
    check_negative_generators_are_all_present,
    check_stage_b_distillation_can_converge,
    check_margin_target_is_reachable,
    check_checkpoint_roundtrip,
    check_resume_equivalence,
    check_throughput,
)


def run_all(verbose: bool = True) -> list[Result]:
    results: list[Result] = []
    for fn in CHECKS:
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:
            r = Result(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
        r.seconds = time.perf_counter() - t0
        results.append(r)
        if verbose:
            tag = "ADVISORY" if r.advisory else ("PASS" if r.ok else "FAIL")
            print(f"[{tag:>8}] {r.name}  ({r.seconds:.1f}s)")
            print(f"           {r.detail}")
    return results


def main() -> int:
    print("SENTRY training pre-flight")
    print("=" * 78)
    results = run_all()

    failed = [r for r in results if not r.ok and not r.advisory]
    print("=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")

    if failed:
        print("\nDo not start a long run.  Failing:")
        for r in failed:
            print(f"  - {r.name}\n      {r.detail}")
        return 1

    print(
        "\nAll invariants hold.  These test the training machinery on a toy\n"
        "model, not pi_0 -- but the machinery is what breaks, and it breaks the\n"
        "same way at both scales.  Before a real run, also confirm Diagnostic\n"
        "D1 on the actual checkpoint (python -m sentry.eval.d1): SS2.8 treats\n"
        "d_prune << d_stale as a gate on the whole approach, and it needs only\n"
        "forward passes -- no training at all."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
