"""Real LIBERO demonstrations, in the form SENTRY's operators consume.

Diagnostic D1 does not need a simulator.  Of its three statistics (SS3.10),
``d_prune`` is measured under a *single* observation, and ``d_stale`` needs
``(o_t, o_{t+k})`` pairs drawn from a real trajectory -- which is exactly what a
demonstration episode is.  Only the N1 scene-perturbation generator and the
final rollout evaluation need to re-render, and those come later.

This reads the LeRobot conversion of LIBERO that ``pi0_libero`` was trained on
(``physical-intelligence/libero``: Spatial + Object + Goal + Long, 1693
episodes, 40 tasks), one parquet per episode:

===============  =====================================================
``image``        PNG bytes, 256x256x3, third-person
``wrist_image``  PNG bytes, 256x256x3
``state``        float[8]   raw proprioception
``actions``      float[7]   raw, gripper at index 6 in +-1
``task_index``   int        joins to ``meta/tasks.jsonl``
===============  =====================================================

Everything the model sees must be built the way openpi builds it, because the
target was trained under those transforms and SS3.9 (ii) makes the normalisation
part of the acceptance rule rather than a preprocessing detail:

- **state** is normalised against the ``state`` statistics over its 8 real
  channels and *then* zero-padded to ``d_a``.  Normalising after padding would
  divide the pad by the pad's own (degenerate) scale.
- **actions** use the ``actions`` statistics -- a different block of the same
  file.  Mixing the two is silent and puts every distance in eq. 10 on the wrong
  footing.
- **images** are ``[-1, 1]``, and resizing goes through openpi's own
  ``resize_with_pad`` rather than a plain interpolate, so a frame reaches the
  encoder the same way it did in training.
- **prompt** is tokenised as pi_0 does it: ``encode(text, add_bos=True)`` plus a
  separately encoded ``"\\n"`` as the start-of-answer token, padded to
  ``max_token_len`` with zeros.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.core.types import ChannelSpec, Observation

__all__ = [
    "LiberoPrompt",
    "LiberoEpisode",
    "LiberoCorpus",
    "state_spec_from_openpi_norm_stats",
    "DELTA_ACTION_DIMS",
    "demo_chunk_to_model_space",
]


DELTA_ACTION_DIMS = 6
"""How many action channels ``pi0_libero`` stores relative to the state.

openpi trains this checkpoint with ``extra_delta_transform=True`` and the mask
``make_bool_mask(6, -1)``: the six pose channels are stored as
``action - state`` (the state at the chunk's *first* frame, broadcast over the
horizon), while the gripper stays absolute.

**Nothing in the checkpoint records this** -- it lives only in the training
config -- and both directions of the mistake are silent:

- forget it at *inference* and the policy emits smooth, plausible actions in the
  wrong frame; measured on LIBERO-Spatial that is 0% success instead of 93%;
- forget it when *building training chunks* and the normalised demonstration
  lands ~8.5 sigma from the origin on channel 3, so every interpolant the loss
  is evaluated on sits far outside the region the policy actually operates in.

The tell is in ``norm_stats``: ``actions/mean`` comes out as very nearly *minus*
``state/mean`` (channel 3: -2.972 against +2.972), which is what statistics over
``action - state`` look like.
"""


def demo_chunk_to_model_space(
    actions: Tensor,
    anchor_state: Tensor,
    spec: ChannelSpec,
    d_a: Optional[int] = None,
) -> Tensor:
    """A demonstration chunk in the policy's normalised action space.

    Args:
        actions: ``(H, 7)`` raw demonstration actions.
        anchor_state: ``(>=6,)`` raw proprioception at the chunk's first frame.
        spec: built from the policy's own ``norm_stats``.
        d_a: padded width; defaults to ``spec.d_a``.

    Applies openpi's ``DeltaActions`` and then normalises, in that order --
    which is the order training used, since the delta transform runs before
    ``Normalize``.  See :data:`DELTA_ACTION_DIMS`.
    """
    d_a = d_a or spec.d_a
    n = actions.shape[-1]
    delta = actions.clone().float()
    k = min(DELTA_ACTION_DIMS, n)
    # One state per chunk, broadcast over the horizon: openpi expands the state
    # along the action axis rather than pairing it frame by frame.
    delta[:, :k] -= anchor_state[:k].float()

    out = torch.zeros(actions.shape[0], d_a, dtype=torch.float32)
    out[:, :n] = (delta - spec.mean[:n]) / spec.scale[:n]
    return out


# --------------------------------------------------------------------------
# Normalisation: the *state* block, which is not the actions block
# --------------------------------------------------------------------------


def state_spec_from_openpi_norm_stats(
    path: str | Path, d_state: int = 8, use_quantiles: bool = False
) -> tuple[Tensor, Tensor]:
    """``(mean, scale)`` over the first ``d_state`` channels of the state stats.

    Returned separately from :class:`ChannelSpec` because a spec describes the
    *action* channels -- their semantics, the gripper sign test, the thresholds
    eq. 12 calibrates.  Proprioception shares none of that; it only has to be
    presented to the policy the way the policy was trained to receive it.
    """
    node = json.loads(Path(path).read_text())
    node = node.get("norm_stats", node)
    if "state" not in node:
        raise ValueError(f"{path} has no 'state' entry; keys: {sorted(node)}")
    st = node["state"]

    if use_quantiles:
        q01 = torch.as_tensor(st["q01"], dtype=torch.float32)[:d_state]
        q99 = torch.as_tensor(st["q99"], dtype=torch.float32)[:d_state]
        mean, scale = (q01 + q99) / 2.0, (q99 - q01) / 2.0
    else:
        mean = torch.as_tensor(st["mean"], dtype=torch.float32)[:d_state]
        scale = torch.as_tensor(st["std"], dtype=torch.float32)[:d_state]

    # openpi divides by ``std + 1e-6``; a channel with no variation would
    # otherwise divide by zero.  Unit scale is the same choice made for the
    # action padding in ``libero_spec.from_openpi_norm_stats``.
    scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
    return mean, scale


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


class LiberoPrompt:
    """pi_0's prompt tokenisation, reproduced exactly.

    ``PaligemmaTokenizer`` in openpi pulls its sentencepiece model from
    ``gs://big_vision/paligemma_tokenizer.model``; point ``model_path`` at a
    local copy of that file.
    """

    def __init__(self, model_path: str | Path, max_len: int = 48) -> None:
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.max_len = max_len
        # Encoded once: pi_0 appends "\n" as a separate start-of-answer token
        # rather than letting it merge into the instruction's last piece.
        self._eol = self._sp.encode("\n")

    def __call__(self, prompt: str) -> Tensor:
        """``(max_len,)`` int64 token ids, zero-padded."""
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._sp.encode(text, add_bos=True) + self._eol
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


def _decode_png(raw: bytes) -> Tensor:
    """PNG bytes -> ``(3, H, W)`` float32 in ``[-1, 1]``."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


@lru_cache(maxsize=1)
def _resize_fn():
    """openpi's own resize, so a frame reaches the encoder as it did in training."""
    from openpi.shared import image_tools

    return image_tools.resize_with_pad_torch


def _resize(img: Tensor, size: int = 224) -> Tensor:
    """``(3,H,W)`` -> ``(3,size,size)`` through openpi's resize_with_pad.

    Fed as an explicit ``(1,3,H,W)`` batch.  ``resize_with_pad_torch`` infers the
    layout from ``shape[-1] <= 4``, which is ambiguous for a bare ``(3,H,W)``
    tensor and decides differently depending on whether a batch dimension is
    present -- so we pin the interpretation rather than rely on the guess.
    """
    if img.shape[-2:] == (size, size):
        return img
    out = _resize_fn()(img.unsqueeze(0), size, size)   # (1,3,H,W) -> (1,3,s,s)
    if out.dim() == 4:
        out = out[0]
    if out.shape[0] != 3:  # came back channels-last
        out = out.permute(2, 0, 1)
    return out.contiguous()


@dataclass
class LiberoEpisode:
    """One demonstration, decoded lazily."""

    index: int
    prompt: str
    images: list[bytes]
    wrist: list[bytes]
    state: Tensor
    """``(T, 8)`` raw proprioception."""
    actions: Tensor
    """``(T, 7)`` raw actions."""

    def __len__(self) -> int:
        return self.state.shape[0]

    def observation(
        self,
        t: int,
        tokens: Tensor,
        state_mean: Tensor,
        state_scale: Tensor,
        d_a: int = 32,
        image_size: int = 224,
    ) -> Observation:
        """Frame ``t`` as a SENTRY :class:`Observation`.

        Two cameras in ``IMAGE_KEYS`` order (base, left wrist); pi_0's third
        slot is filled and masked off by the backend, which is what
        ``LiberoInputs`` does for ``right_wrist_0_rgb``.
        """
        if not (0 <= t < len(self)):
            raise IndexError(f"frame {t} outside [0, {len(self)})")

        imgs = torch.stack(
            [
                _resize(_decode_png(self.images[t]), image_size),
                _resize(_decode_png(self.wrist[t]), image_size),
            ]
        )

        raw = self.state[t]
        norm = (raw - state_mean) / state_scale
        padded = torch.zeros(d_a, dtype=torch.float32)
        padded[: norm.shape[0]] = norm

        return Observation(images=imgs, language=tokens, state=padded, t=t)


class LiberoCorpus:
    """A directory of LeRobot LIBERO episodes.

    ``root`` holds ``data/chunk-*/episode_*.parquet`` and ``meta/episodes.jsonl``
    (which carries each episode's task text, so ``task_index`` never has to be
    guessed).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))
        if not self.paths:
            raise FileNotFoundError(f"no episode parquets under {self.root}/data")

        meta = self.root / "meta" / "episodes.jsonl"
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta} missing -- it carries the task text per episode; "
                "fetch it from the dataset repo's meta/ directory"
            )
        self.prompts: dict[int, str] = {}
        for line in meta.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tasks = rec.get("tasks") or []
            if tasks:
                self.prompts[int(rec["episode_index"])] = tasks[0]

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, i: int) -> LiberoEpisode:
        import pyarrow.parquet as pq

        path = self.paths[i]
        table = pq.read_table(path)
        ep_index = int(table.column("episode_index")[0].as_py())
        prompt = self.prompts.get(ep_index)
        if prompt is None:
            raise KeyError(f"no task text for episode {ep_index} in meta/episodes.jsonl")

        return LiberoEpisode(
            index=ep_index,
            prompt=prompt,
            images=[r["bytes"] for r in table.column("image").to_pylist()],
            wrist=[r["bytes"] for r in table.column("wrist_image").to_pylist()],
            state=torch.tensor(
                [list(x) for x in table.column("state").to_pylist()], dtype=torch.float32
            ),
            actions=torch.tensor(
                [list(x) for x in table.column("actions").to_pylist()], dtype=torch.float32
            ),
        )

    def episodes(self, n: Optional[int] = None) -> Sequence[LiberoEpisode]:
        return [self.load(i) for i in range(min(n or len(self), len(self)))]
