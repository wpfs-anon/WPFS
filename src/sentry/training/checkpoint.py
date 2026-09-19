"""Checkpointing built for an environment that will interrupt you.

Table 2 puts Stage A at 20k steps and Stage B at 60k.  A Colab session caps at
roughly 12 hours and reclaims an idle runtime after about 90 minutes, so a run
of that length *will* be interrupted -- resume is not a nicety to add later,
it is part of the design.

Two properties matter and neither is free:

- **Atomic writes.**  The checkpoint goes to a temporary file and is then
  renamed.  A crash mid-write otherwise leaves a truncated file where the
  previous good checkpoint used to be, which is the worst possible outcome:
  you lose the run *and* the ability to resume it.  This matters more on Drive
  than on local disk, where writes are slower and more likely to be cut short.
- **Resume equivalence.**  Restoring must reproduce the uninterrupted run.
  Optimiser state is the usual omission -- AdamW's moments are most of what
  makes training work, and a resume that drops them looks fine (loss continues
  to fall) while quietly restarting the optimiser's warm-up.
  :mod:`sentry.training.preflight` tests this rather than assuming it.
"""

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
    """Write a checkpoint atomically.

    Stores only the adapter tensors and the optimiser state -- ``theta`` is
    frozen throughout (SS2.5), so persisting the base weights would multiply
    the file size for nothing and invite a checkpoint that silently disagrees
    with the pretrained policy it is supposed to leave untouched.
    """
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
    os.replace(tmp, path)          # atomic on POSIX and on Windows
    return path


def load(path: str | Path, trainer: Any, strict_stage: Optional[str] = None) -> dict:
    """Restore a checkpoint into ``trainer``.  Returns the ``extra`` payload."""
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
    """Find the highest-step checkpoint in ``directory``.

    Used on Colab restart: point at the Drive folder and continue from wherever
    the previous session was cut off.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    best: Optional[CheckpointInfo] = None
    for p in directory.glob("*.pt"):
        try:
            head = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue  # a truncated file from a crash mid-write; skip it
        if stage is not None and head.get("stage") != stage:
            continue
        info = CheckpointInfo(path=p, step=int(head.get("step", 0)), stage=head.get("stage", "?"))
        if best is None or info.step > best.step:
            best = info
    return best
