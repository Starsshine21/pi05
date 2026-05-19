from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi06_recap.labels import EpisodeOutcome, nttg_value, value_to_bin


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


@dataclass(frozen=True)
class ManifestConfig:
    lerobot_root: Path
    image_key: str | None = None
    max_steps: int = 500
    num_bins: int = 201
    sample_stride: int = 1
    default_prompt: str = ""
    failed_value: float = -1.0


def read_tasks(lerobot_root: str | Path) -> dict[int, str]:
    """Best-effort parser for LeRobot task metadata."""
    root = Path(lerobot_root)
    meta = root / "meta"
    tasks: dict[int, str] = {}

    jsonl = meta / "tasks.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row.get("task_index", row.get("index", len(tasks))))
            text = row.get("task") or row.get("prompt") or row.get("instruction") or row.get("name")
            if text:
                tasks[idx] = str(text)

    json_path = meta / "tasks.json"
    if json_path.exists() and not tasks:
        raw = json.loads(json_path.read_text())
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    text = value.get("task") or value.get("prompt") or value.get("instruction") or value.get("name")
                else:
                    text = value
                if text is not None:
                    tasks[int(key)] = str(text)
        elif isinstance(raw, list):
            for idx, value in enumerate(raw):
                if isinstance(value, dict):
                    task_idx = int(value.get("task_index", idx))
                    text = value.get("task") or value.get("prompt") or value.get("instruction") or value.get("name")
                else:
                    task_idx = idx
                    text = value
                if text is not None:
                    tasks[task_idx] = str(text)

    return tasks


def detect_image_key(lerobot_root: str | Path) -> str:
    root = Path(lerobot_root)
    candidates = sorted((root / "videos").glob("chunk-*/*"))
    for path in candidates:
        if path.is_dir() and path.name.startswith("observation.images."):
            return path.name
    raise FileNotFoundError(
        f"Could not detect a video camera key under {root / 'videos'}. "
        "Pass --image-key explicitly, for example observation.images.top."
    )


def data_files(lerobot_root: str | Path) -> list[Path]:
    root = Path(lerobot_root)
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    return files


def episode_index_from_file(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def find_video_path(lerobot_root: str | Path, image_key: str, parquet_file: Path, episode_index: int) -> Path:
    root = Path(lerobot_root)
    chunk = parquet_file.parent.name
    direct = root / "videos" / chunk / image_key / f"episode_{episode_index:06d}.mp4"
    if direct.exists():
        return direct

    matches = sorted((root / "videos").glob(f"chunk-*/{image_key}/episode_{episode_index:06d}.mp4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No video found for episode {episode_index} camera {image_key} under {root / 'videos'}")


def _not_missing(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        import pandas as pd

        return bool(pd.notna(value))
    except Exception:
        return True


def _task_text(row: Any, tasks: dict[int, str], default_prompt: str) -> str:
    for col in ("prompt", "task", "instruction", "language_instruction"):
        if col in row and _not_missing(row[col]):
            return str(row[col])
    if "task_index" in row and _not_missing(row["task_index"]):
        task = tasks.get(int(row["task_index"]))
        if task:
            return task
    return default_prompt


def build_manifest_rows(config: ManifestConfig, outcomes: dict[int, EpisodeOutcome]) -> list[dict[str, Any]]:
    import pandas as pd

    root = config.lerobot_root
    image_key = config.image_key or detect_image_key(root)
    tasks = read_tasks(root)
    rows: list[dict[str, Any]] = []

    for parquet_file in data_files(root):
        episode_index = episode_index_from_file(parquet_file)
        outcome = outcomes.get(episode_index)
        if outcome is None:
            raise KeyError(
                f"Episode {episode_index} is missing from outcomes. "
                "The success file must contain every episode used for VF training."
            )

        df = pd.read_parquet(parquet_file)
        if df.empty:
            continue
        frame_col = "frame_index" if "frame_index" in df.columns else None
        last_frame = int(df[frame_col].max()) if frame_col else len(df) - 1
        video_path = find_video_path(root, image_key, parquet_file, episode_index)

        for local_idx, row in df.iterrows():
            frame_index = int(row[frame_col]) if frame_col else int(local_idx)
            if config.sample_stride > 1 and frame_index % config.sample_stride != 0:
                continue
            value = nttg_value(
                frame_index=frame_index,
                outcome=outcome,
                episode_last_frame=last_frame,
                max_steps=config.max_steps,
                failed_value=config.failed_value,
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": float(row["timestamp"]) if "timestamp" in row and pd.notna(row["timestamp"]) else None,
                    "prompt": _task_text(row, tasks, config.default_prompt),
                    "image_key": image_key,
                    "video_path": str(video_path),
                    "video_frame": frame_index,
                    "value": value,
                    "value_bin": value_to_bin(value, num_bins=config.num_bins),
                    "success": outcome.success,
                }
            )
    return rows


def write_jsonl(rows: list[dict[str, Any]], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
