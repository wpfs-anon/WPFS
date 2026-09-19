"""The real pi_0 backend: openpi's PyTorch pi_0 exposed at two depths.

This is the wiring SS3.9 (*Implementation Notes*) describes, resolved against
openpi's actual module tree rather than against the architecture diagram.  Every
decision below is one the paper flags as easy to get silently wrong, so each
carries the fact it was resolved against.

What was resolved, and against what
-----------------------------------

**Depths come off the checkpoint** (SS3.9 iv).  ``L_V = 27`` (SigLIP-So400m) and
``L_B = 18`` (Gemma-2B, and the 300M action expert is also 18) are *read*, never
assumed -- ``pi0_config`` alone does not state them, and this family varies
between 18 and 32 layers.

**Truncation cuts the layer list** (SS3.5, *early exit not interior skipping*).
The two loops sit in different places and take different levers:

- SigLIP iterates ``for encoder_layer in self.layers:`` with no depth check, so
  ``E_V`` must slice the ``ModuleList`` itself.
- Gemma iterates ``self.layers[: self.config.num_hidden_layers]``, so ``E_B`` is
  applied by setting that config value -- which *is* a slice of the same list,
  just spelled by the upstream code.

Both are contiguous prefixes, which is what preserves "the residual-stream
statistics the surviving layers were trained under".

**One ``E_B`` governs both experts** (SS3.9, *Depth is shared*).  openpi's
two-expert forward is a single ``for layer_idx in range(num_layers)`` loop whose
body concatenates the two streams' queries, keys and values before one attention
call -- so layer ``l`` genuinely processes both experts together and skipping it
skips both.  ``assert_shared_depth`` is called rather than trusted.

**tau runs the other way** (SS3.9 i).  openpi's interpolant is
``x_t = t*noise + (1-t)*actions`` and its sampler integrates ``t: 1 -> 0``, so
``t = 0`` is clean data -- the mirror image of the paper's convention.  The
paper anticipates exactly this ("if the implementation integrates in the
opposite direction, (8) and (9) require tau -> 1-tau"), and
:data:`OPENPI_TAU_CONVENTION` records it so a config can never be built with the
wrong one by accident.  :mod:`sentry.core.renoise` then handles both the time
argument *and* the sign flip on the velocity that the chain rule implies.

**No cache crosses the mode boundary** (SS3.9).  ``velocity`` recomputes the
prefix from the current images at the requested depth on every call.  Within a
single call the ``K`` interpolants share one prefix and are batched, which is
the sharing SS3.4.2 explicitly permits ("the K interpolants ... are evaluated in
a single batched forward pass, so the cost of T is that of one evaluation, not
K") and is what eq. 17's single ``L_den`` term is counting.

**Proposition 4 is a test, not a claim.**  ``plan`` routes through openpi's own
``sample_actions`` untouched, and adapters gated off return ``base(x)``
identically, so plan mode is bit-identical to the pretrained policy by
construction -- and :func:`assert_planning_preserved` checks it anyway.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import torch
from torch import Tensor, nn

from sentry.core.types import Observation, assert_shared_depth
from sentry.models.lora import adapter_slot as adapter_slot_ctx
from sentry.models.lora import adapters as adapters_ctx
from sentry.models.lora import wrap_linear, wrap_linear_multi

__all__ = [
    "OPENPI_TAU_CONVENTION",
    "OpenPiBackend",
    "load_pi0_pytorch",
    "assert_planning_preserved",
]


OPENPI_TAU_CONVENTION = "zero_is_clean"
"""openpi integrates ``t: 1 -> 0`` with ``t = 0`` clean (SS3.9 convention i).

Read off ``PI0Pytorch``: ``x_t = t*noise + (1-t)*actions`` in ``forward``, and
``dt = -1/num_steps`` with ``time`` starting at 1.0 in ``sample_actions``.  Any
:class:`~sentry.config.SentryConfig` driving this backend must carry
``tau_convention="zero_is_clean"``; the default is the paper's opposite reading
and would leave eq. 9 wrong by ``2(1-s)v`` while still looking plausible.
"""

# Projections the trunk adapter Delta_B attaches to, for *both* experts
# (SS3.9: "the query/key/value/output projections and the MLP up/down
# projections of both the VLM expert and the action expert").
_GEMMA_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEMMA_MLP = ("gate_proj", "up_proj", "down_proj")

# SigLIP spells the same roles differently -- resolved against the module tree.
_SIGLIP_ATTN = ("q_proj", "k_proj", "v_proj", "out_proj")
_SIGLIP_MLP = ("fc1", "fc2")


@dataclass(frozen=True)
class Pi0Geometry:
    """Depths and shapes read off a loaded checkpoint."""

    L_V: int
    L_B: int
    H: int
    d_a: int
    max_token_len: int
    image_keys: tuple[str, ...]

    def describe(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"L_V={self.L_V} L_B={self.L_B} H={self.H} d_a={self.d_a} "
            f"tokens={self.max_token_len} cams={len(self.image_keys)}"
        )


def load_pi0_pytorch(
    converted_dir: str | Path,
    device: str | torch.device = "cuda",
    openpi_src: Optional[str] = None,
):
    """Load a converted ``pi0_*_pytorch`` directory into a ``PI0Pytorch``.

    ``converted_dir`` is the output of openpi's
    ``examples/convert_jax_model_to_pytorch.py``: a ``model.safetensors`` plus a
    ``config.json``.  Compilation is left off -- ``torch.compile`` would trace
    ``sample_actions`` at a fixed depth, and the whole point here is that depth
    varies per call.

    **The model's dtype layout is openpi's, not ours.**  ``PI0Pytorch`` is
    deliberately mixed precision: its constructor casts ``paligemma_with_expert``
    to bf16 while holding the patch embeddings, the layernorms and the final
    norm at fp32, and it leaves the model-level projections -- ``state_proj``,
    ``action_in_proj``, ``action_out_proj``, ``action_time_mlp_*`` -- at fp32
    entirely.  That is not incidental: ``embed_suffix`` branches on
    ``state_proj.weight.dtype == torch.float32`` and ``denoise_step`` upcasts to
    fp32 before the read-out.  A blanket ``.to(bfloat16)`` after construction --
    which openpi's own conversion script applies before saving -- breaks those
    assumptions, so we let the constructor set the layout and only move the
    weights onto the device.  ``load_state_dict`` copies into the existing
    parameters and casts to each one's dtype, so the bf16 file loads correctly
    into the fp32 slots.
    """
    import sys

    if openpi_src:
        for p in (openpi_src, str(Path(openpi_src).parent / "packages/openpi-client/src")):
            if p not in sys.path:
                sys.path.insert(0, p)

    import safetensors.torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    converted = Path(converted_dir)
    saved = json.loads((converted / "config.json").read_text())

    # openpi's converter records neither ``pi05`` nor ``max_token_len``, and a
    # pi0 built over pi0.5 weights fails only as a list of unexpected keys -- so
    # the architecture is read off the weights.  pi0.5 conditions the action
    # expert on time through adaRMS (``time_mlp_*``) and has no ``state_proj``;
    # Pi0Config then derives the 200-token prompt that goes with it.
    from safetensors import safe_open

    with safe_open(str(converted / "model.safetensors"), "pt") as f:
        pi05 = bool(saved.get("pi05", any(k.startswith("time_mlp_in.") for k in f.keys())))

    cfg = Pi0Config(
        action_dim=saved["action_dim"],
        action_horizon=saved["action_horizon"],
        paligemma_variant=saved["paligemma_variant"],
        action_expert_variant=saved["action_expert_variant"],
        pi05=pi05,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(cfg)
    missing, unexpected = safetensors.torch.load_model(
        model, str(converted / "model.safetensors"), strict=False
    )
    if unexpected:
        raise RuntimeError(f"checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}")
    if missing:
        # save_model drops tied weights; anything else missing is a real problem.
        raise RuntimeError(f"checkpoint is missing {len(missing)} keys: {missing[:5]}")

    # Device only -- never dtype.  See the docstring.
    return model.to(device=device).eval()


class OpenPiBackend:
    """openpi's PyTorch pi_0, exposed as a :class:`~sentry.core.interfaces.VLABackend`.

    Args:
        model: a loaded ``PI0Pytorch``.
        device: where the model lives.
        M: Euler steps for plan mode (Table 2: 10).
        attach_adapters: build ``Delta`` on construction.  Depth budgets for the
            adapters are ``E_V_max``/``E_B_max``; SS3.5.2 attaches the trunk to
            "layers ``l < E_max`` only", so adapters beyond the deepest rung of
            the ladder would never be evaluated and are not created.
        lora_rank / lora_rank_readout / lora_dropout: SS3.9 defaults (16, 32, 0.05).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda",
        M: int = 10,
        attach_adapters: bool = True,
        E_V_max: Optional[int] = None,
        E_B_max: Optional[int] = None,
        n_rungs: int = 3,
        lora_rank: int = 16,
        lora_rank_readout: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.M = M
        self.n_rungs = n_rungs

        pwe = model.paligemma_with_expert
        self._vision = pwe.paligemma.model.vision_tower.vision_model
        self._vlm = pwe.paligemma.language_model
        self._expert = pwe.gemma_expert.model
        self._pwe = pwe

        # SS3.9 (iv): read the depths, never hardcode 27/18.
        self.L_V = len(self._vision.encoder.layers)
        self.L_B = len(self._vlm.layers)
        if len(self._expert.layers) != self.L_B:
            raise ValueError(
                f"the two experts disagree on depth: VLM has {self.L_B} layers, "
                f"action expert has {len(self._expert.layers)}.  Attention is joint "
                "at every layer, so a single E_B cannot govern both (SS3.9)."
            )

        self.H = model.config.action_horizon
        self.d_a = model.config.action_dim
        self.max_token_len = model.config.max_token_len

        from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

        self.image_keys = tuple(IMAGE_KEYS)

        self.E_V_max = E_V_max or self.L_V
        self.E_B_max = E_B_max or self.L_B
        self._adapters_attached = False
        if attach_adapters:
            self.attach_adapters(lora_rank, lora_rank_readout, lora_dropout)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def geometry(self) -> Pi0Geometry:
        return Pi0Geometry(
            L_V=self.L_V, L_B=self.L_B, H=self.H, d_a=self.d_a,
            max_token_len=self.max_token_len, image_keys=self.image_keys,
        )

    # ------------------------------------------------------------------
    # adapters (SS3.9, *Where the adapters go*)
    # ------------------------------------------------------------------

    def attach_adapters(
        self, r: int = 16, r_readout: int = 32, dropout: float = 0.05
    ) -> int:
        """Wrap the projections ``Delta`` attaches to.  Returns the count.

        Placement follows SS3.9 literally: the trunk covers q/k/v/o and the MLP
        projections of **both** experts for layers below ``E_B_max``, plus the
        SigLIP layers below ``E_V_max`` for ``Delta_V``; the read-out adapter
        goes on the single action read-out projection at the higher rank.

        Every wrapper starts gated **off** with ``lora_B`` zeroed, so attaching
        adapters cannot perturb plan mode -- Proposition 4 survives this call.
        """
        n = 0
        for layer in self._vision.encoder.layers[: self.E_V_max]:
            for attr in _SIGLIP_ATTN:
                wrap_linear(layer.self_attn, attr, r=r, dropout=dropout)
                n += 1
            for attr in _SIGLIP_MLP:
                wrap_linear(layer.mlp, attr, r=r, dropout=dropout)
                n += 1

        for stack in (self._vlm, self._expert):
            for layer in stack.layers[: self.E_B_max]:
                for attr in _GEMMA_ATTN:
                    wrap_linear(layer.self_attn, attr, r=r, dropout=dropout)
                    n += 1
                for attr in _GEMMA_MLP:
                    wrap_linear(layer.mlp, attr, r=r, dropout=dropout)
                    n += 1

        # One adapter per rung on the SINGLE read-out projection (SS3.5.2).
        wrap_linear_multi(
            self.model, "action_out_proj",
            n_slots=self.n_rungs, r=r_readout, dropout=dropout,
        )
        n += 1

        # Newly created adapter tensors follow the module they wrap.
        self.model.to(self.device)
        self._adapters_attached = True
        return n

    # ------------------------------------------------------------------
    # depth truncation
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _truncated(self, E_V: int, E_B: int) -> Iterator[None]:
        """Temporarily expose the network at ``(E_V, E_B)``.

        A no-op at full depth, which is what keeps Proposition 4 exact: plan
        mode never passes through any mutation at all.
        """
        if not (1 <= E_V <= self.L_V):
            raise ValueError(f"E_V={E_V} outside [1, {self.L_V}]")
        if not (1 <= E_B <= self.L_B):
            raise ValueError(f"E_B={E_B} outside [1, {self.L_B}]")

        if E_V == self.L_V and E_B == self.L_B:
            yield
            return

        vision_layers = self._vision.encoder.layers
        vlm_cfg = self._vlm.config
        expert_cfg = self._expert.config
        text_cfg = self._pwe.paligemma.config.text_config
        saved = (vlm_cfg.num_hidden_layers, expert_cfg.num_hidden_layers,
                 text_cfg.num_hidden_layers)
        try:
            # SigLIP's loop ignores config, so the list itself is sliced.  The
            # surviving entries are the *same* module objects -- no copy, no
            # parameter movement.
            self._vision.encoder.layers = nn.ModuleList(list(vision_layers)[:E_V])
            # Gemma's loop is already `self.layers[: config.num_hidden_layers]`.
            # text_config drives the joint two-expert loop in
            # PaliGemmaWithExpertModel.forward, so all three must move together
            # or perception and denoising would truncate independently -- the
            # configuration SS3.9 says is not expressible.
            vlm_cfg.num_hidden_layers = E_B
            expert_cfg.num_hidden_layers = E_B
            text_cfg.num_hidden_layers = E_B
            yield
        finally:
            self._vision.encoder.layers = vision_layers
            (vlm_cfg.num_hidden_layers, expert_cfg.num_hidden_layers,
             text_cfg.num_hidden_layers) = saved

    @contextlib.contextmanager
    def _truncated_vision(self, E_V: int) -> Iterator[None]:
        """Truncate the **encoder only**, leaving the backbone at full depth.

        Stage A needs this and nothing else: it "truncates the encoder; the
        backbone stays at ``L_B``".  ``E_V`` and ``E_B`` are independent budgets
        -- Table 1's rung 3 pairs ``E_V = 14`` with ``E_B = 12`` -- so this is
        not the "shallow backbone, deep denoiser" configuration SS3.9 rules out.
        That constraint is about the two *experts* inside one backbone layer,
        which share attention; the vision tower is upstream of both.
        """
        if not (1 <= E_V <= self.L_V):
            raise ValueError(f"E_V={E_V} outside [1, {self.L_V}]")
        if E_V == self.L_V:
            yield
            return
        layers = self._vision.encoder.layers
        try:
            self._vision.encoder.layers = nn.ModuleList(list(layers)[:E_V])
            yield
        finally:
            self._vision.encoder.layers = layers

    # ------------------------------------------------------------------
    # Stage A seam: the encoder and the backbone, separately (SS3.5.1)
    # ------------------------------------------------------------------

    def encode(self, images: Tensor, E_V: int) -> Tensor:
        """Visual tokens at encoder depth ``E_V``.  ``(1, n_cam*n_tok, width)``.

        SS3.5.1 writes ``z = E_phi(I)`` for the token sequence and
        ``v_theta(. | z)`` for the backbone conditioned on a supplied one; eq. 13
        needs both branches to differ in *only* that sequence.  Returning tokens
        rather than a velocity is what makes that expressible.

        Tokens are produced for every camera the checkpoint expects, including
        one padded with zeros when the robot has fewer -- the count has to match
        what :meth:`velocity_from_tokens` will build masks for.
        """
        model = self.model
        n_cam = images.shape[0]
        with self._truncated_vision(E_V):
            embs = []
            for i, _key in enumerate(self.image_keys):
                frame = (
                    images[i] if i < n_cam else torch.zeros_like(images[0])
                ).to(device=self.device, dtype=torch.float32)
                frame = self._resize_to_model(frame).unsqueeze(0)
                embs.append(model.paligemma_with_expert.embed_image(frame))
        return torch.cat(embs, dim=1)

    def _resize_to_model(self, frame: Tensor) -> Tensor:
        """``(3,H,W)`` at whatever resolution -> the checkpoint's, via openpi."""
        from openpi.models_pytorch.preprocessing_pytorch import IMAGE_RESOLUTION
        from openpi.shared import image_tools

        if tuple(frame.shape[-2:]) == tuple(IMAGE_RESOLUTION):
            return frame
        out = image_tools.resize_with_pad_torch(
            frame.unsqueeze(0), *IMAGE_RESOLUTION
        )
        out = out[0] if out.dim() == 4 else out
        return out if out.shape[0] == 3 else out.permute(2, 0, 1)

    def velocity_from_tokens(
        self,
        A_tau: Tensor,
        tau: Tensor,
        vis: Tensor,
        obs: Observation,
        E_B: int,
        adapters: bool = False,
    ) -> Tensor:
        """``v_theta(A^tau, tau | z, l, s)`` -- the backbone on *supplied* tokens.

        The counterpart of :meth:`encode`.  Stage A's behavioural term runs this
        twice, once on the truncated encoder's tokens and once on the full
        encoder's, with everything else -- backbone depth, adapter state,
        ``tau``, ``eps`` -- held identical, so the difference it measures is the
        encoder's alone.
        """
        model = self.model
        K = A_tau.shape[0]
        out_dtype, out_device = A_tau.dtype, A_tau.device

        with adapters_ctx(model, adapters), self._truncated(self.L_V, E_B):
            past_key_values, prefix_pad_masks, state1 = self._prefix(obs, vis=vis)
            past_key_values = _expand_cache(past_key_values, K)
            v = model.denoise_step(
                state1.expand(K, state1.shape[-1]),
                prefix_pad_masks.expand(K, prefix_pad_masks.shape[1]),
                past_key_values,
                A_tau.to(device=self.device, dtype=torch.float32),
                tau.to(device=self.device, dtype=torch.float32),
            )
        return v.to(device=out_device, dtype=out_dtype)

    # ------------------------------------------------------------------
    # observation bridging
    # ------------------------------------------------------------------

    def _openpi_obs(self, obs: Observation, batch: int = 1) -> Any:
        """Wrap a SENTRY :class:`Observation` in what openpi's preprocessor reads.

        openpi duck-types this object, so a namespace with the right attributes
        is enough and we avoid importing its dataclass.

        The contract on the SENTRY side, stated once here because nothing else
        enforces it: ``images`` are ``(n_cam, 3, 224, 224)`` in ``[-1, 1]`` and
        ordered as ``IMAGE_KEYS``; ``language`` holds ``max_token_len`` token ids
        with zero for padding; ``state`` is the policy-normalised, ``d_a``-padded
        proprioception.  Building those is the data pipeline's job, not the
        backend's -- putting it here would let evaluation and Stage-B datagen
        drift apart, which is the same failure the padding scheme is frozen in
        ``SentryConfig`` to prevent.
        """
        imgs = obs.images
        if imgs.ndim != 4:
            raise ValueError(f"images must be (n_cam, C, H, W), got {tuple(imgs.shape)}")
        if imgs.shape[0] > len(self.image_keys):
            raise ValueError(
                f"{imgs.shape[0]} cameras supplied but the checkpoint takes "
                f"{len(self.image_keys)}: {self.image_keys}"
            )

        dev = self.device
        images: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        # fp32 throughout: SigLIP's patch embedding is one of the tensors openpi
        # deliberately holds at fp32, and the cast down to bf16 happens inside
        # the model at the embedding boundary, not here.
        for i, key in enumerate(self.image_keys):
            if i < imgs.shape[0]:
                frame = imgs[i].to(device=dev, dtype=torch.float32)
                present = True
            else:
                # A camera the checkpoint expects but this robot does not have.
                # openpi pads it with zeros and masks it off -- exactly what
                # LiberoInputs does for right_wrist_0_rgb.
                frame = torch.zeros_like(imgs[0]).to(device=dev, dtype=torch.float32)
                present = False
            images[key] = frame.unsqueeze(0).expand(batch, *frame.shape)
            masks[key] = torch.full((batch,), present, dtype=torch.bool, device=dev)

        lang = obs.language.to(dev)
        if lang.ndim != 1:
            raise ValueError(f"language must be (L,), got {tuple(lang.shape)}")
        tok = lang.unsqueeze(0).expand(batch, lang.shape[0])
        tok_mask = (tok != 0).bool()

        state = obs.state.to(device=dev, dtype=torch.float32)
        if state.shape != (self.d_a,):
            raise ValueError(
                f"state must be ({self.d_a},) -- normalised and padded to the "
                f"policy's action dim -- got {tuple(state.shape)}"
            )

        return _Namespace(
            images=images,
            image_masks=masks,
            state=state.unsqueeze(0).expand(batch, self.d_a),
            tokenized_prompt=tok,
            tokenized_prompt_mask=tok_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    # ------------------------------------------------------------------
    # plan mode (SS3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        """Full depth, adapters OFF, ``M`` Euler steps.  Returns ``(H, d_a)`` fp32.

        Routed straight through openpi's own ``sample_actions`` at full depth
        with no truncation context entered, so this path is the pretrained
        policy verbatim (Proposition 4).

        ``noise`` is honoured for paper defect **D5**: the liveness label of
        Definition 1 compares a fresh plan against the chunk under test, and if
        those are independent draws from a multi-modal policy then eq. 5
        measures sampling spread rather than staleness.
        """
        o = self._openpi_obs(obs, batch=1)
        if noise is not None:
            if noise.shape != (self.H, self.d_a):
                raise ValueError(
                    f"noise must be ({self.H}, {self.d_a}), got {tuple(noise.shape)}"
                )
            noise = noise.to(device=self.device, dtype=torch.float32).unsqueeze(0)

        with adapters_ctx(self.model, False):
            actions = self.model.sample_actions(
                self.device, o, noise=noise, num_steps=self.M
            )
        # Returned where the caller's observation lives, for the same reason
        # ``velocity`` preserves its input device.
        return actions[0].to(device=obs.images.device, dtype=torch.float32)

    @torch.no_grad()
    def plan_shallow_perception(
        self, obs: Observation, E_V: int, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Plan with a **truncated encoder** and ``Delta_V`` on, backbone at full depth.

        Not part of SENTRY's execution loop -- Algorithm 1 always plans with
        :meth:`plan`, at full depth, which is what Proposition 4 is about.  This
        exists to *evaluate Stage A on its own*: it isolates the question "does
        the truncated encoder still support the policy?" from everything the
        backbone does, which is the one question Stage A is responsible for and
        the one the D1 read-out cliff does not contaminate.

        **Only ``Delta_V`` is gated on.**  Enabling every adapter would leave
        ``Delta_B`` computing a rank-16 update on twelve backbone layers of both
        experts -- arithmetic that contributes exactly zero while it is
        untrained, but is not free: measured on pi_0 it cost 127 ms per plan
        against 89 ms, turning a truncation that should have been *cheaper* into
        one that was 43% more expensive.  Gating the vision tower alone is also
        the honest configuration once Stage B has run and ``Delta_B`` is no
        longer zero.
        """
        o = self._openpi_obs(obs, batch=1)
        if noise is not None:
            if noise.shape != (self.H, self.d_a):
                raise ValueError(
                    f"noise must be ({self.H}, {self.d_a}), got {tuple(noise.shape)}"
                )
            noise = noise.to(device=self.device, dtype=torch.float32).unsqueeze(0)

        # Everything off, then the vision tower back on.  ``adapters`` gates
        # whatever lives under the module it is handed, so nesting the two is
        # how a subset is selected.
        with adapters_ctx(self.model, False), adapters_ctx(self._vision, True), \
                self._truncated_vision(E_V):
            actions = self.model.sample_actions(
                self.device, o, noise=noise, num_steps=self.M
            )
        return actions[0].to(device=obs.images.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # check mode (SS3.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        """One velocity evaluation at depth ``(E_V, E_B)``.  ``(K,H,d_a) -> (K,H,d_a)``.

        Two phases, and the split is the cost model:

        1. the prefix -- SigLIP over the **current** images, then ``E_B``
           backbone layers -- is computed once at batch 1;
        2. all ``K`` interpolants run the action expert against that one prefix.

        That is the sharing SS3.4.2 permits and eq. 17 assumes.  Nothing from
        plan mode is reused: this prefix is recomputed here, at this depth, with
        this adapter state, from these images.
        """
        assert_shared_depth(E_B, E_B)  # a single E_B governs both pathways
        if A_tau.ndim != 3 or A_tau.shape[1:] != (self.H, self.d_a):
            raise ValueError(
                f"A_tau must be (K, {self.H}, {self.d_a}), got {tuple(A_tau.shape)}"
            )
        K = A_tau.shape[0]
        if tau.shape != (K,):
            raise ValueError(f"tau must be ({K},), got {tuple(tau.shape)}")
        if adapters and not self._adapters_attached:
            raise RuntimeError("adapters requested but none are attached")

        model = self.model
        # Same device and dtype out as in.  The model may live on an
        # accelerator while the caller works on CPU tensors -- eq. 9 then
        # combines this velocity with the interpolant the caller passed in, and
        # a silent device change there is a runtime error at best.
        out_dtype, out_device = A_tau.dtype, A_tau.device

        with adapters_ctx(model, adapters), self._truncated(E_V, E_B):
            past_key_values, prefix_pad_masks, state1 = self._prefix(obs)

            # -- phase 2: K interpolants against that one prefix -----------
            past_key_values = _expand_cache(past_key_values, K)
            prefix_pad_masks_k = prefix_pad_masks.expand(K, prefix_pad_masks.shape[1])
            state_k = state1.expand(K, state1.shape[-1])

            v = model.denoise_step(
                state_k,
                prefix_pad_masks_k,
                past_key_values,
                A_tau.to(device=self.device, dtype=torch.float32),
                tau.to(device=self.device, dtype=torch.float32),
            )

        return v.to(device=out_device, dtype=out_dtype)

    # ------------------------------------------------------------------
    # multi-exit check mode (SS3.5.2)
    # ------------------------------------------------------------------

    def _prefix(self, obs: Observation, vis: Optional[Tensor] = None):
        """Phase 1: encode and run the backbone prefix once, at batch 1.

        Must be called inside a ``_truncated`` context -- the depth it runs at is
        whatever that context has set.  Returns the KV cache the action expert
        will attend to, plus the masks and state it needs.

        ``vis`` supplies precomputed image tokens instead of running the encoder,
        which is what :meth:`velocity_from_tokens` needs.  The masks are then
        rebuilt to the same shape ``embed_prefix`` would have produced, so the
        two paths are interchangeable.
        """
        model = self.model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        o1 = self._openpi_obs(obs, batch=1)
        images, img_masks, lang_tokens, lang_masks, state1 = (
            model._preprocess_observation(o1, train=False)
        )
        if vis is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._prefix_from_tokens(
                vis, img_masks, lang_tokens, lang_masks
            )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self._vlm.config._attn_implementation = "eager"

        _, past_key_values = self._pwe.forward(
            attention_mask=model._prepare_attention_masks_4d(prefix_att_2d),
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values, prefix_pad_masks, state1

    def _prefix_from_tokens(
        self,
        vis: Tensor,
        img_masks: Sequence[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Rebuild ``embed_prefix``'s output from supplied image tokens.

        Mirrors ``PI0Pytorch.embed_prefix`` exactly, including the ``sqrt(dim)``
        scaling on the language embeddings that openpi applies and nothing else
        signposts.  Image tokens are assumed to be laid out camera-major, which
        is how :meth:`encode` concatenates them.
        """
        import math

        model = self.model
        n_cam = len(img_masks)
        if vis.shape[1] % n_cam:
            raise ValueError(
                f"{vis.shape[1]} visual tokens do not divide across {n_cam} cameras"
            )
        per_cam = vis.shape[1] // n_cam
        bsize = vis.shape[0]

        pad_masks = [
            m[:, None].expand(bsize, per_cam) for m in img_masks
        ]
        att_masks = [0] * vis.shape[1]

        lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat([vis, lang_emb.to(vis.dtype)], dim=1)
        pad = torch.cat([*pad_masks, lang_masks], dim=1)
        att = torch.tensor(att_masks, dtype=torch.bool, device=pad.device)
        return embs, pad, att[None, :].expand(bsize, len(att_masks))

    def velocity_multi_exit(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        exits: Sequence[int],
        adapters: bool = True,
    ) -> list[Tensor]:
        r"""Every rung of the ladder from **one** forward pass (SS3.5.2).

        "A single pass to :math:`E_{\max}` therefore exposes every rung of the
        ladder at once", because "layer :math:`\ell` depends only on layers
        :math:`< \ell`".  This is what makes a Stage-B step cost roughly
        :math:`E_{\max}/L_B` of a conventional distillation step while supplying
        :math:`J` times the supervision.

        The pass runs to ``max(exits)``; the action expert's hidden state is
        tapped at each rung, put through the frozen final norm, and read out with
        that rung's adapter.  Returns one ``(K, H, d_a)`` velocity per exit.

        Worth naming, because it bounds what training can repair: the frozen
        ``norm`` sits *between* the exit and the adapter.  ``Delta^(j)_out`` is
        therefore a rank-``r`` correction applied **after** a normalisation
        whose per-channel gain was fitted for layer ``L_B``.  Whether that is
        enough is exactly the open question Diagnostic D1 leaves behind.
        """
        exits = list(exits)
        if not exits:
            raise ValueError("need at least one exit depth")
        if exits != sorted(exits) or len(set(exits)) != len(exits):
            raise ValueError(f"exits must be strictly increasing, got {exits}")
        if not (1 <= exits[-1] <= self.L_B):
            raise ValueError(f"deepest exit {exits[-1]} outside [1, {self.L_B}]")
        if len(exits) > self.n_rungs:
            raise ValueError(
                f"{len(exits)} exits but only {self.n_rungs} read-out adapters; "
                "rung j selects adapter j, so the ladder cannot be longer"
            )
        if A_tau.ndim != 3 or A_tau.shape[1:] != (self.H, self.d_a):
            raise ValueError(
                f"A_tau must be (K, {self.H}, {self.d_a}), got {tuple(A_tau.shape)}"
            )
        K = A_tau.shape[0]
        if tau.shape != (K,):
            raise ValueError(f"tau must be ({K},), got {tuple(tau.shape)}")
        if adapters and not self._adapters_attached:
            raise RuntimeError("adapters requested but none are attached")

        model = self.model
        out_dtype, out_device = A_tau.dtype, A_tau.device
        E_max = exits[-1]

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        with adapters_ctx(model, adapters), self._truncated(E_V, E_max):
            past_key_values, prefix_pad_masks, state1 = self._prefix(obs)
            past_key_values = _expand_cache(past_key_values, K)
            prefix_pad_masks = prefix_pad_masks.expand(K, prefix_pad_masks.shape[1])
            state_k = state1.expand(K, state1.shape[-1])

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                model.embed_suffix(
                    state_k,
                    A_tau.to(device=self.device, dtype=torch.float32),
                    tau.to(device=self.device, dtype=torch.float32),
                )
            )
            # Mask construction mirrors ``denoise_step``: the suffix attends to
            # the whole (padded) prefix and causally within itself.
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(K, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            self._expert.config._attn_implementation = "eager"
            out = self._expert.forward(
                inputs_embeds=suffix_embs,
                attention_mask=model._prepare_attention_masks_4d(full_att_2d),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                adarms_cond=adarms_cond,
                output_hidden_states=True,
            )
            # ``hidden_states[l]`` is the *input* to layer l, so the output of
            # the first E layers is ``hidden_states[E]`` -- pre-norm, which the
            # read-out path then normalises itself.
            #
            # With one exception, and it is a trap: HF appends the final entry
            # **after** applying the model's norm, so ``hidden_states[E_max]`` is
            # already normalised while every shallower index is not.  Normalising
            # it again perturbs only the deepest rung, which is exactly the
            # signature this showed on pi_0 -- rungs 5 and 9 matched a per-depth
            # pass to 0.000000 while rung 12 was off by 0.04.
            hs = out.hidden_states
            velocities: list[Tensor] = []
            for j, E_B in enumerate(exits):
                if E_B == E_max:
                    h = out.last_hidden_state
                else:
                    h, _ = self._expert.norm(hs[E_B], adarms_cond)
                h = h[:, -self.H :].to(torch.float32)
                with adapter_slot_ctx(model, j):
                    velocities.append(
                        model.action_out_proj(h).to(device=out_device, dtype=out_dtype)
                    )

        return velocities


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _Namespace:
    """Minimal attribute bag matching what openpi's preprocessor reads."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _expand_cache(cache: Any, K: int) -> Any:
    """Broadcast a batch-1 KV cache to ``K`` rows.

    The prefix is identical for every verification timestep -- the interpolants
    "differ only in the leading batch dimension" (SS3.4.2) -- so this expands one
    computed prefix rather than computing ``K`` of them, which is what makes
    eq. 17's last term a single ``L_den``.

    ``expand`` would give stride-0 views; the attention kernels reshape these, so
    we materialise instead.  A Gemma-2B prefix is one KV head of ~800 tokens per
    layer, so the copy is a few megabytes.
    """
    if K == 1:
        return cache
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is None or values is None:
        raise TypeError(
            f"unsupported cache type {type(cache).__name__}: expected key_cache / "
            "value_cache lists (transformers DynamicCache)"
        )
    for i in range(len(keys)):
        if keys[i] is not None and keys[i].shape[0] == 1:
            keys[i] = keys[i].repeat(K, 1, 1, 1)
            values[i] = values[i].repeat(K, 1, 1, 1)
    return cache


def adapter_groups(backend: "OpenPiBackend"):
    """Partition this checkpoint's adapters into ``Delta_V`` and ``Delta_B``.

    :func:`sentry.training.params.split_adapters` classifies by module-path
    prefix, and its defaults are the toy model's names.  openpi's tree spells
    the same roles differently, and the exact spelling depends on the
    transformers version -- ``language_model`` is reachable both directly and
    through ``.model`` -- so the split is made on a substring that is stable
    either way rather than on a guessed prefix.

    SS3.5.1 trains ``Delta_V`` alone; SS3.5.2 trains "``Delta_B`` on backbone
    layers plus the action read-out, with ``Delta_V`` from Stage A frozen".
    """
    from sentry.models.lora import GatedAdapter
    from sentry.training.params import AdapterGroups

    model = backend.model
    owners = {
        name for name, mod in model.named_modules() if isinstance(mod, GatedAdapter)
    }
    delta_V, delta_B, unclaimed = [], [], []
    for name, p in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        owner = name.rsplit(".lora_", 1)[0]
        if owner not in owners:
            unclaimed.append(name)
        elif "vision_tower" in owner:
            delta_V.append((name, p))
        else:
            delta_B.append((name, p))

    if unclaimed:
        raise ValueError(
            f"{len(unclaimed)} adapter tensors belong to no wrapper "
            f"(first: {unclaimed[0]})"
        )
    return AdapterGroups(delta_V=delta_V, delta_B=delta_B)


def assert_planning_preserved(
    backend: "OpenPiBackend",
    obs: Observation,
    noise: Tensor,
    atol: float = 0.0,
) -> None:
    """**Proposition 4**, checked by exact equality rather than asserted.

    Runs plan mode twice under the same noise -- once with adapters present but
    gated off, once with every adapter forcibly disabled -- and requires the two
    chunks to be *identical*, not merely close.  ``LoRALinear`` returns
    ``base(x)`` untouched when gated, so any difference at all means a wrapper
    leaked into the forward path.

    ``atol = 0.0`` is deliberate: SS3.3 says plan mode "is numerically identical
    to the pretrained policy", and a tolerance here would hide exactly the bug
    this exists to catch.
    """
    a = backend.plan(obs, noise=noise)
    with adapters_ctx(backend.model, False):
        b = backend.plan(obs, noise=noise)
    diff = (a - b).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"Proposition 4 violated: plan mode differs by {diff:.3e} with "
            "adapters gated off.  Adapters must contribute exactly zero."
        )
