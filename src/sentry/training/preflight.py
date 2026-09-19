"""Pre-flight checks -- run these before committing a GPU to a long run.

Every check here targets a failure that is **silent**: training proceeds, the
loss goes down, and the resulting adapters are wrong.  A loss curve cannot tell
you that Stage B quietly re-tuned ``Delta_V``, that the teacher branch was
never detached, that ``tau`` was drawn from ``U(0,1)`` instead of the deployed
schedule, or that resume dropped the optimiser moments.  You find out days
later, from a verifier that accepts everything.

The checks run on CPU in about a minute, against
:class:`sentry.models.toy_pi0.TinyPi0`.  They test the **training machinery**,
not pi_0 -- but the machinery is what breaks, and it breaks identically at both
scales.

Run: ``python -m sentry.training.preflight``
"""

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
    """Advisory checks report a number to judge, not a pass/fail invariant."""


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _rig(H: int = 8):
    """A small but structurally faithful setup: two experts, shared attention."""
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


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_adapter_partition() -> Result:
    """``Delta_V`` and ``Delta_B`` must be disjoint and cover every adapter.

    An adapter in neither group is trained by no stage and stays at its zero
    initialisation -- inert, and invisible unless you count tensors.
    """
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
    """Stage A trains ``Delta_V``.  "Nothing else." (SS2.5.1)

    Two warm-up steps first: LoRA starts with ``B = 0``, so ``dL/dA`` is
    exactly zero on the first backward and the audit would read every
    ``lora_A`` as idle for reasons that have nothing to do with wiring.
    """
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
    """Stage B trains ``Delta_B`` + read-out; ``Delta_V`` is frozen (SS2.5.2).

    The leak this is really watching for is ``Delta_V`` picking up gradient:
    Stage B silently re-tuning the encoder undoes the perception warm-start
    Stage A spent 20k steps on, and the loss curve says nothing about it.
    """
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    rng = random.Random(0)
    for _ in range(2):
        tr.step(batch, rng)

    tr.opt.zero_grad(set_to_none=True)
    # Audit at the deepest rung of the curriculum, where the most adapters are
    # active; shallower draws legitimately leave the upper layers idle.
    loss, _ = tr.loss_on(batch, rng, torch.Generator().manual_seed(0), E_B=model.L_B)
    loss.backward()
    rep = audit(model, "Stage B", tr.groups.delta_B)
    return Result("Stage B gradient audit (Delta_V frozen)", rep.ok, str(rep))


def check_theta_never_moves() -> Result:
    """``theta`` is frozen throughout -- Proposition 4 depends on it."""
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
    """After training, gating adapters off must be **bit-identical** again.

    Proposition 4 is claimed "for any ``Delta`` obtained by the training of
    Sec. 2.5", so it has to survive an actual optimiser step, not merely hold
    at initialisation where ``B`` is still zero.
    """
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
    """The teacher sits under ``sg[.]``, so it must carry no graph.

    Beyond correctness this is roughly half the activation memory -- on a 16 GB
    T4 it decides whether pi_0 fits at all.
    """
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
    """Stage B draws ``tau ~ T``, "not ``U(0,1)``" (SS2.5.2 step 4).

    The shallow mode is only ever queried at the timesteps in ``T``; training
    it across the whole path spends capacity where it will never be asked.
    """
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
    """``E_B ~ p_depth`` must actually vary (SS2.5.2, *Depth curriculum*).

    "A single ``Delta_B`` then serves a *range* of depths, which is what makes
    the depth cascade of Sec. 2.6 possible without training one adapter per
    depth."  A curriculum stuck on one depth trains an adapter that only works
    at that rung, and the cascade silently degrades.
    """
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
    """Equation 15's two unstated edge cases.

    ``h_star == H_k``: "The second term is omitted" -- there is no invalid
    position, and including it would train the model to reject the padded tail
    that eq. 11 never evaluates.  ``h_star == 0``: the sum is empty and
    ``1/h_star`` would divide by zero.
    """
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
    """Equation 16 must penalise the degenerate optimum it exists to exclude.

    "A degenerate optimum of equation 14 alone is a shallow mode that ignores
    ``o*`` and reproduces ``A_hat`` by copying the interpolant, which yields
    ``d_h ~ 0`` everywhere and a verifier that accepts unconditionally."
    """
    eta = 0.05
    collapsed = torch.randn(4, 8, 7)
    penalty_collapsed = sensitivity_loss(collapsed, collapsed.clone(), eta)
    penalty_healthy = sensitivity_loss(collapsed, collapsed + 5.0, eta)

    # Tolerance, not equality: eta round-trips through float32.
    ok = abs(float(penalty_collapsed) - eta) < 1e-6 and float(penalty_healthy) == 0.0
    return Result(
        "eq. 16 penalises an observation-ignoring verifier",
        ok,
        f"collapsed -> {float(penalty_collapsed):.4f} (want {eta}), "
        f"sensitive -> {float(penalty_healthy):.4f} (want 0)",
    )


def check_finite_and_deterministic() -> Result:
    """No NaN/Inf, and the same seed reproduces the same loss."""
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
    """The single most informative smoke test for any training loop.

    If the loss will not fall on one batch held fixed, no amount of data or
    GPU-hours will help -- something is disconnected.
    """
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
    """``plan(obs, noise)`` must be a function of ``(obs, noise)`` alone.

    **Paper defect D5.**  Definition 1 writes ``pi_theta(o_{t+k})`` as though it
    were a value, but a flow-matching policy returns a *sample* from a
    multi-modal distribution -- the reason the target is a diffusion policy at
    all.  Comparing two independent draws makes eq. 5 measure the policy's own
    sampling spread rather than staleness, and that label is the foundation of
    both eq. 12's calibration and eq. 15's ``h_star``.

    On this untrained model the spread is ~4.0 per position against
    ``liveness_eps = 0.05``, i.e. two hundred thousand times the tolerance it is
    compared against -- which is how every sample in a Stage-B batch came to be
    labelled stale at position 0.  The fix is a common random number, and this
    asserts the backend honours it.
    """
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
    """``h_star`` must not be degenerate across a Stage-B batch.

    This is the check that would have caught D5.  Seventeen other checks passed
    while every sample in the batch carried ``h_star = 0`` -- they all verified
    *machinery* (gradients reach the right tensors, the teacher is detached,
    ``tau`` comes from the deployed schedule) and none verified that the labels
    those mechanisms consume mean anything.

    Three invariants, all from SS2.5.2, stated per generator rather than in
    aggregate:

    - **positives satisfy ``h_star == H_k``.**  A positive is one whose scene
      "has evolved only through the robot's own execution of the plan", so it is
      live by construction and has no invalid position to push up.  Any other
      value hands eq. 15 a push-up target on a live plan, putting it in direct
      opposition to eq. 14.  This is the invariant D5 was breaking.
    - **N4 satisfies ``h_star >= 1``.**  Its corruption starts at ``h_0`` drawn
      from ``{1, ..., H_k-1}``, and the prefix before it is untouched -- so a
      zero here means the ``h_0`` it supplies by construction is being
      overwritten by something else.
    - **the batch carries more than one distinct ``h_star``.**  The degenerate
      case that actually occurred was *every* label at 0.

    Deliberately **not** asserted: that N1/N3 give ``h_star > 0``.  A displaced
    or swapped target invalidates the plan from the very first action, so
    ``h_star = 0`` is the *correct* label there, and demanding otherwise would
    be demanding the label be wrong.
    """
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
    """A Stage-B batch must be able to produce all four of N1-N4.

    SS2.5.2: negatives are "drawn uniformly from four generators".  The recipe
    was implemented correctly in :mod:`sentry.eval.harness` and then
    reimplemented -- N1 only -- in :mod:`sentry.training.datagen`, so Stage B
    trained on a quarter of it for as long as that lasted.  What was lost was
    specific: N4 is "the only generator that supplies a supervised target for
    the accepted *length* rather than for the binary label", and it is the only
    source of ``h_star`` that does not route through ``liveness`` -- hence the
    only one immune to D5.

    Drawn over a batch large enough that all four are near-certain, then
    asserted rather than assumed.

    ``j_min`` is lowered from its default of 5 for this fixture.  N2 pairs the
    candidate with an observation ``j`` steps away, ``|j| > j_min``, and needs
    ``0 <= k + j < H``; at the toy's ``H = 8`` the default leaves almost no
    admissible ``(k, j)``, so N2 would be missing for a reason that is about the
    fixture rather than about the generator.  The default suits ``H = 50``.
    """
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
    """Can ``L_dist`` fall at all, with the competing terms switched off?

    Stage A had an overfit check from the start and Stage B did not, which is
    how a non-converging ``L_dist`` went unnoticed.  This isolates the
    machinery: with ``lambda_m = lambda_s = 0`` the objective is exactly
    equation 14, and if the shallow mode cannot be pulled onto the full-depth
    teacher on a single fixed batch at a fixed depth, something is wired wrong.
    Weighting is judged separately, by the advisory check below.
    """
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
    """Does the **full-depth teacher** satisfy equation 15's hinge?

    Equation 14 pulls the student onto the teacher; equation 15 reshapes a
    thresholded functional of the student's reconstruction.  If the teacher's
    own ``L_marg`` is large, the two are asking for different things and
    ``lambda_m * L_marg`` will fight ``L_dist`` -- the student is being told to
    outperform the very target it is being distilled from.

    Advisory, because the honest reading depends on the checkpoint.  On an
    untrained model the velocity field is meaningless, so the teacher may fail
    the hinge for reasons that say nothing about a real policy.  **Run this
    against pi_0** before trusting Table 2's ``lambda_m = 1.0``: a teacher that
    passes its own hinge makes the two terms compatible, and a teacher that
    does not makes them rivals.

    History worth keeping, because it changes how to read a ``RIVALS`` verdict.
    This check was written after ``L_dist`` was observed not to converge, and it
    reported ``RIVALS`` -- which was taken as evidence that eq. 14 and eq. 15 are
    structurally opposed.  They are not.  The teacher was failing a hinge posed
    against ``h_star = 0`` on **every** sample, positives included, because the
    liveness label was comparing two independent draws of a stochastic policy
    (paper defect D5, see
    :func:`check_plan_is_deterministic_under_shared_noise`).  With the label
    fixed the hinge is posed against the right position.  So: if this reports
    ``RIVALS``, check
    :func:`check_liveness_labels_are_informative` **first** -- a bad label
    produces the same symptom as a genuine conflict, and only one of the two is
    a finding.
    """
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
    """Save and load must restore adapters **and** optimiser state.

    Dropping AdamW's moments is the classic silent resume bug: the loss keeps
    falling, so nothing looks wrong, while the optimiser restarts its warm-up.
    """
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
    """An interrupted-and-resumed run must equal the uninterrupted one.

    This is the property Colab will exercise for you whether or not you test
    it.  Ten steps straight through, versus five, checkpoint, five more.
    """
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
    """Project Table 2's 20k / 60k steps onto measured step time.

    Advisory: the number is for this toy model on CPU and says nothing about
    pi_0 on a GPU.  What it *does* tell you is the shape of the arithmetic --
    at 20k + 60k steps, a step time of 0.5 s is 11 hours, which is already past
    a Colab session cap and means resume must work before you start.
    """
    model, cfg, spec, env_cfg = _rig()
    tr = _stage_b(model, cfg, spec)
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=3), random.Random(0)
    )
    rng = random.Random(0)
    tr.step(batch, rng)                                   # warm-up
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
    # -- the data the objectives consume, before the objectives themselves --
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
        except Exception as exc:  # a check that crashes is a failure, not a stop
            r = Result(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
        r.seconds = time.perf_counter() - t0
        results.append(r)
        if verbose:
            tag = "ADVISORY" if r.advisory else ("PASS" if r.ok else "FAIL")
            print(f"[{tag:>8}] {r.name}  ({r.seconds:.1f}s)")
            print(f"           {r.detail}")
    return results


def main() -> int:  # pragma: no cover - reporting
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
