from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def threshold_for_group(values: list[float], positive_fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    positive_fraction = max(1e-6, min(1.0, positive_fraction))
    q = 1.0 - positive_fraction
    values_sorted = sorted(values)
    idx = int(math.ceil(q * len(values_sorted)))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return values_sorted[idx]


def _dense_rewards_from_targets(rows: list[dict]) -> list[float]:
    rewards = [0.0] * len(rows)
    for i, row in enumerate(rows):
        target = row.get("target_value")
        if target is None:
            raise KeyError("nstep mode requires 'target_value' in score rows. Re-run score_vf.py on a labeled manifest.")
        current = float(target)
        if i + 1 < len(rows) and int(rows[i + 1]["episode_index"]) == int(row["episode_index"]):
            next_target = rows[i + 1].get("target_value")
            if next_target is None:
                raise KeyError("nstep mode requires 'target_value' in every score row.")
            rewards[i] = current - float(next_target)
        else:
            rewards[i] = current
    return rewards


def _n_step_advantages(rows: list[dict], *, n_step: int) -> list[float]:
    if n_step <= 0:
        raise ValueError("n_step must be > 0")
    rewards = _dense_rewards_from_targets(rows)
    values = [float(row["vf_score"]) for row in rows]
    advantages = [0.0] * len(rows)
    for i, row in enumerate(rows):
        episode = int(row["episode_index"])
        total = 0.0
        j = i
        steps = 0
        while steps < n_step and j < len(rows) and int(rows[j]["episode_index"]) == episode:
            total += rewards[j]
            j += 1
            steps += 1
        bootstrap = values[j] if steps == n_step and j < len(rows) and int(rows[j]["episode_index"]) == episode else 0.0
        advantages[i] = total + bootstrap - values[i]
    return advantages


def label_scores(
    rows: list[dict],
    *,
    mode: str,
    threshold: float,
    positive_fraction: float,
    group_by_prompt: bool,
    n_step: int,
) -> tuple[list[dict], dict[str, float]]:
    rows = sorted(rows, key=lambda r: (int(r["episode_index"]), int(r["frame_index"])))
    score_field = "vf_score"
    if mode == "nstep":
        advantages = _n_step_advantages(rows, n_step=n_step)
        for row, adv in zip(rows, advantages, strict=True):
            row["advantage_score"] = float(adv)
        score_field = "advantage_score"
        mode = "quantile"

    thresholds: dict[str, float] = {}
    if mode == "threshold":
        thresholds["__global__"] = threshold
    elif mode == "quantile":
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            key = row.get("prompt", "__global__") if group_by_prompt else "__global__"
            groups[key].append(float(row[score_field]))
        thresholds = {key: threshold_for_group(values, positive_fraction) for key, values in groups.items()}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    labeled: list[dict] = []
    for row in rows:
        key = row.get("prompt", "__global__") if group_by_prompt else "__global__"
        row_threshold = thresholds.get(key, thresholds.get("__global__", threshold))
        score = float(row[score_field])
        label = "positive" if score >= row_threshold else "negative"
        labeled.append(
            {
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "advantage": label,
                "advantage_text": f"Advantage: {label}",
                "vf_score": float(row["vf_score"]),
                "advantage_score": score,
                "threshold": float(row_threshold),
                "prompt": row.get("prompt", ""),
            }
        )
    return labeled, thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VF scores to positive/negative advantage labels.")
    parser.add_argument("--scores", required=True, help="JSONL from score_vf.py")
    parser.add_argument("--output", required=True, help="JSONL lookup consumed by the openpi prompt patch.")
    parser.add_argument("--mode", choices=["nstep", "threshold", "quantile"], default="nstep")
    parser.add_argument("--n-step", type=int, default=50, help="N-step horizon over manifest rows for Evo-RL-style ACP.")
    parser.add_argument("--threshold", type=float, default=-0.2, help="Used when --mode threshold.")
    parser.add_argument("--positive-fraction", type=float, default=0.30, help="Top fraction kept positive in quantile mode.")
    parser.add_argument("--global-threshold", dest="group_by_prompt", action="store_false")
    parser.set_defaults(group_by_prompt=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.scores)
    labeled, thresholds = label_scores(
        rows,
        mode=args.mode,
        threshold=args.threshold,
        positive_fraction=args.positive_fraction,
        group_by_prompt=args.group_by_prompt,
        n_step=args.n_step,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_pos = sum(row["advantage"] == "positive" for row in labeled)
    print(f"Wrote {len(labeled):,} labels to {output}")
    print(f"positive={n_pos:,} negative={len(labeled) - n_pos:,} thresholds={len(thresholds)}")


if __name__ == "__main__":
    main()
