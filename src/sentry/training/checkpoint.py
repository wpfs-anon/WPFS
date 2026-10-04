from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

__all__ = ["save", "load", "latest", "CheckpointInfo"]


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    stage: str


def save(
    path: str | Path,
    trainer: Any,
    stage: str,
    extra: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": stage,
        "step": trainer.step_idx,
        "trainer": trainer.state_dict(),
        "extra": extra or {},
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load(path: str | Path, trainer: Any, strict_stage: Optional[str] = None) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    if strict_stage is not None and payload["stage"] != strict_stage:
        raise ValueError(
            f"checkpoint is from stage {payload['stage']!r}, expected "
            f"{strict_stage!r}.  Loading Stage-A adapters into a Stage-B "
            "trainer would restore Delta_V into Delta_B's slots."
        )

    trainer.load_state_dict(payload["trainer"])
    return payload.get("extra", {})


def latest(directory: str | Path, stage: Optional[str] = None) -> Optional[CheckpointInfo]:
    directory = Path(directory)
    if not directory.is_dir():
        return None

    best: Optional[CheckpointInfo] = None
    for p in directory.glob("*.pt"):
        try:
            head = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if stage is not None and head.get("stage") != stage:
            continue
        info = CheckpointInfo(path=p, step=int(head.get("step", 0)), stage=head.get("stage", "?"))
        if best is None or info.step > best.step:
            best = info
    return best
