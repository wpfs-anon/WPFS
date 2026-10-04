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

_GEMMA_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_GEMMA_MLP = ("gate_proj", "up_proj", "down_proj")

_SIGLIP_ATTN = ("q_proj", "k_proj", "v_proj", "out_proj")
_SIGLIP_MLP = ("fc1", "fc2")


@dataclass(frozen=True)
class Pi0Geometry:

    L_V: int
    L_B: int
    H: int
    d_a: int
    max_token_len: int
    image_keys: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"L_V={self.L_V} L_B={self.L_B} H={self.H} d_a={self.d_a} "
            f"tokens={self.max_token_len} cams={len(self.image_keys)}"
        )


def load_pi0_pytorch(
    converted_dir: str | Path,
    device: str | torch.device = "cuda",
    openpi_src: Optional[str] = None,
):
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
        raise RuntimeError(f"checkpoint is missing {len(missing)} keys: {missing[:5]}")

    return model.to(device=device).eval()


class OpenPiBackend:

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


    @property
    def geometry(self) -> Pi0Geometry:
        return Pi0Geometry(
            L_V=self.L_V, L_B=self.L_B, H=self.H, d_a=self.d_a,
            max_token_len=self.max_token_len, image_keys=self.image_keys,
        )


    def attach_adapters(
        self, r: int = 16, r_readout: int = 32, dropout: float = 0.05
    ) -> int:
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

        wrap_linear_multi(
            self.model, "action_out_proj",
            n_slots=self.n_rungs, r=r_readout, dropout=dropout,
        )
        n += 1

        self.model.to(self.device)
        self._adapters_attached = True
        return n


    @contextlib.contextmanager
    def _truncated(self, E_V: int, E_B: int) -> Iterator[None]:
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
            self._vision.encoder.layers = nn.ModuleList(list(vision_layers)[:E_V])
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


    def encode(self, images: Tensor, E_V: int) -> Tensor:
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


    def _openpi_obs(self, obs: Observation, batch: int = 1) -> Any:
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
        for i, key in enumerate(self.image_keys):
            if i < imgs.shape[0]:
                frame = imgs[i].to(device=dev, dtype=torch.float32)
                present = True
            else:
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


    @torch.no_grad()
    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
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
        return actions[0].to(device=obs.images.device, dtype=torch.float32)

    @torch.no_grad()
    def plan_shallow_perception(
        self, obs: Observation, E_V: int, noise: Optional[Tensor] = None
    ) -> Tensor:
        o = self._openpi_obs(obs, batch=1)
        if noise is not None:
            if noise.shape != (self.H, self.d_a):
                raise ValueError(
                    f"noise must be ({self.H}, {self.d_a}), got {tuple(noise.shape)}"
                )
            noise = noise.to(device=self.device, dtype=torch.float32).unsqueeze(0)

        with adapters_ctx(self.model, False), adapters_ctx(self._vision, True), \
                self._truncated_vision(E_V):
            actions = self.model.sample_actions(
                self.device, o, noise=noise, num_steps=self.M
            )
        return actions[0].to(device=obs.images.device, dtype=torch.float32)


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
        assert_shared_depth(E_B, E_B)
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

        with adapters_ctx(model, adapters), self._truncated(E_V, E_B):
            past_key_values, prefix_pad_masks, state1 = self._prefix(obs)

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


    def _prefix(self, obs: Observation, vis: Optional[Tensor] = None):
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


class _Namespace:

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _expand_cache(cache: Any, K: int) -> Any:
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
    a = backend.plan(obs, noise=noise)
    with adapters_ctx(backend.model, False):
        b = backend.plan(obs, noise=noise)
    diff = (a - b).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"Proposition 4 violated: plan mode differs by {diff:.3e} with "
            "adapters gated off.  Adapters must contribute exactly zero."
        )
