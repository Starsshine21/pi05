from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_index: int
    success: bool
    success_frame: int | None = None


def value_to_bin(value: float, *, num_bins: int = 201) -> int:
    """Map a normalized value in [-1, 0] to a discrete bin."""
    clipped = max(-1.0, min(0.0, float(value)))
    return int(round((clipped + 1.0) * (num_bins - 1)))


def bin_to_value(bin_id: int, *, num_bins: int = 201) -> float:
    """Map a value bin back to the normalized scalar center in [-1, 0]."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    clipped = max(0, min(num_bins - 1, int(bin_id)))
    return clipped / (num_bins - 1) - 1.0


def nttg_value(
    *,
    frame_index: int,
    outcome: EpisodeOutcome,
    episode_last_frame: int,
    max_steps: int = 500,
    failed_value: float = -1.0,
) -> float:
    """Negative time-to-go value used by the VF reproduction.

    Successful episodes receive 0 at the success frame and linearly decrease
    backwards to -1 over ``max_steps``. Failed episodes receive ``failed_value``.
    Values are normalized to [-1, 0].
    """
    if not outcome.success:
        return max(-1.0, min(0.0, float(failed_value)))

    success_frame = outcome.success_frame
    if success_frame is None:
        success_frame = episode_last_frame
    remaining = max(0, int(success_frame) - int(frame_index))
    return max(-1.0, -remaining / float(max_steps))


def read_outcomes(path: str | Path) -> dict[int, EpisodeOutcome]:
    """Read episode outcomes from CSV or JSONL.

    Required columns/keys:
      - episode_index
      - success

    Optional:
      - success_frame, success_frame_index, or final_success_frame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    outcomes: dict[int, EpisodeOutcome] = {}
    if path.suffix.lower() == ".jsonl":
        rows: Iterable[dict] = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    for row in rows:
        episode_index = int(row["episode_index"])
        success_raw = row.get("success", False)
        if isinstance(success_raw, str):
            success = success_raw.strip().lower() in {"1", "true", "yes", "y", "success", "succeeded"}
        else:
            success = bool(success_raw)

        success_frame_raw = (
            row.get("success_frame")
            or row.get("success_frame_index")
            or row.get("final_success_frame")
            or row.get("terminal_frame")
        )
        success_frame = int(success_frame_raw) if success_frame_raw not in (None, "") else None
        outcomes[episode_index] = EpisodeOutcome(
            episode_index=episode_index,
            success=success,
            success_frame=success_frame,
        )
    return outcomes

