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


def demo_chunk_to_model_space(
    actions: Tensor,
    anchor_state: Tensor,
    spec: ChannelSpec,
    d_a: Optional[int] = None,
) -> Tensor:
    d_a = d_a or spec.d_a
    n = actions.shape[-1]
    delta = actions.clone().float()
    k = min(DELTA_ACTION_DIMS, n)
    delta[:, :k] -= anchor_state[:k].float()

    out = torch.zeros(actions.shape[0], d_a, dtype=torch.float32)
    out[:, :n] = (delta - spec.mean[:n]) / spec.scale[:n]
    return out


def state_spec_from_openpi_norm_stats(
    path: str | Path, d_state: int = 8, use_quantiles: bool = False
) -> tuple[Tensor, Tensor]:
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

    scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
    return mean, scale


class LiberoPrompt:

    def __init__(self, model_path: str | Path, max_len: int = 48) -> None:
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.max_len = max_len
        self._eol = self._sp.encode("\n")

    def __call__(self, prompt: str) -> Tensor:
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._sp.encode(text, add_bos=True) + self._eol
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def _decode_png(raw: bytes) -> Tensor:
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


@lru_cache(maxsize=1)
def _resize_fn():
    from openpi.shared import image_tools

    return image_tools.resize_with_pad_torch


def _resize(img: Tensor, size: int = 224) -> Tensor:
    if img.shape[-2:] == (size, size):
        return img
    out = _resize_fn()(img.unsqueeze(0), size, size)
    if out.dim() == 4:
        out = out[0]
    if out.shape[0] != 3:
        out = out.permute(2, 0, 1)
    return out.contiguous()


@dataclass
class LiberoEpisode:

    index: int
    prompt: str
    images: list[bytes]
    wrist: list[bytes]
    state: Tensor
    actions: Tensor

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
