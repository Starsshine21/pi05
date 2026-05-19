#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import traceback
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency: numpy", file=sys.stderr)
    print("Please run this script in the pi05 environment, for example:", file=sys.stderr)
    print("  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh", file=sys.stderr)
    print("  conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final", file=sys.stderr)
    print("  source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh", file=sys.stderr)
    print("  python scripts/convert_pick_place_to_lerobot.py --overwrite --stride 1 --image-height 224 --image-width 224", file=sys.stderr)
    raise

try:
    from datasets import Dataset
except ModuleNotFoundError:
    print("Missing dependency: datasets", file=sys.stderr)
    print("Please run this script in the pi05 environment, for example:", file=sys.stderr)
    print("  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh", file=sys.stderr)
    print("  conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final", file=sys.stderr)
    print("  source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh", file=sys.stderr)
    print("  python scripts/convert_pick_place_to_lerobot.py --overwrite --stride 1 --image-height 224 --image-width 224", file=sys.stderr)
    raise

from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_hf_features_from_features, write_episode, write_episode_stats, write_info
from PIL import Image
from tqdm import tqdm

FPS = 10
DEFAULT_REPO_ID = "local/pi05-pickplace-il"
REQUIRED_KEYS = [
    "episode_ur5e_pos_j",
    "episode_ur5e_pos_eef",
    "episode_inspire_hand_pos",
    "episode_l515_color",
    "episode_orbbec_femto_bolt_color",
]


def parse_name(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    if len(parts) < 5:
        return {"episode_id": parts[0], "lighting": "unknown", "surface": "unknown", "object": stem, "target": "unknown"}
    return {"episode_id": parts[0], "lighting": parts[1], "surface": parts[2], "object": parts[3], "target": "_".join(parts[4:])}


def make_task_text(meta: dict[str, str]) -> str:
    obj_name = meta["object"].replace("-", " ")
    surface = meta["surface"].replace("-", " ")
    target = meta["target"].replace("_", " ").replace("-", " ")
    return f"place the {obj_name} at the {target} position on the {surface}"


def load_episode(path: Path) -> dict:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    return data


def validate_episode(data: dict, path: Path) -> None:
    joints = np.asarray(data["episode_ur5e_pos_j"])
    eef = np.asarray(data["episode_ur5e_pos_eef"])
    hand = np.asarray(data["episode_inspire_hand_pos"])
    cam0 = np.asarray(data["episode_l515_color"])
    cam1 = np.asarray(data["episode_orbbec_femto_bolt_color"])

    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"Invalid joint shape in {path.name}: {joints.shape}")
    if eef.ndim != 2 or eef.shape[1] != 6:
        raise ValueError(f"Invalid eef shape in {path.name}: {eef.shape}")
    if hand.ndim != 2 or hand.shape[1] < 1:
        raise ValueError(f"Invalid hand shape in {path.name}: {hand.shape}")
    if cam0.ndim != 4 or cam0.shape[-1] != 3:
        raise ValueError(f"Invalid l515 image shape in {path.name}: {cam0.shape}")
    if cam1.ndim != 4 or cam1.shape[-1] != 3:
        raise ValueError(f"Invalid orbbec image shape in {path.name}: {cam1.shape}")

    lengths = [len(joints), len(eef), len(hand), len(cam0), len(cam1)]
    if min(lengths) <= 1:
        raise ValueError(f"Too few frames in {path.name}: {lengths}")
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched trajectory lengths in {path.name}: {lengths}")



def compute_action(data: dict, mode: str, *, use_next_state_action: bool = False) -> np.ndarray:
    joints = np.asarray(data["episode_ur5e_pos_j"], dtype=np.float32)
    eef = np.asarray(data["episode_ur5e_pos_eef"], dtype=np.float32)
    hand = np.asarray(data["episode_inspire_hand_pos"], dtype=np.float32)
    if mode == "joint_delta":
        base = joints
    elif mode == "eef_delta":
        base = np.concatenate([eef, hand], axis=1)
    else:
        base = np.concatenate([joints, hand], axis=1)
    if use_next_state_action:
        action = np.zeros_like(base, dtype=np.float32)
        action[:-1] = base[1:]
        action[-1] = base[-1] if len(base) > 0 else 0
        return action
    action = np.zeros_like(base, dtype=np.float32)
    action[:-1] = base[1:] - base[:-1]
    action[-1] = action[-2] if len(action) > 1 else 0
    return action


def build_features(state_dim: int, action_dim: int, image_size: tuple[int, int], *, include_task: bool = True) -> dict:
    height, width = image_size
    features = {
        "image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "actions": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    if include_task:
        features["prompt"] = {"dtype": "string", "shape": (1,), "names": None}
    return features


def resize_frame(frame: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    target_h, target_w = image_size
    return Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR)


def build_columns(data: dict, action_mode: str, episode_index: int, task: str, task_index: int, global_offset: int, stride: int, image_size: tuple[int, int], *, use_next_state_action: bool = False) -> dict[str, list]:
    joints = np.asarray(data["episode_ur5e_pos_j"], dtype=np.float32)
    eef = np.asarray(data["episode_ur5e_pos_eef"], dtype=np.float32)
    hand = np.asarray(data["episode_inspire_hand_pos"], dtype=np.float32)
    cam0 = np.asarray(data["episode_l515_color"], dtype=np.uint8)
    cam1 = np.asarray(data["episode_orbbec_femto_bolt_color"], dtype=np.uint8)
    action = compute_action(data, action_mode, use_next_state_action=use_next_state_action).astype(np.float32)
    state = np.concatenate([joints, eef, hand], axis=1).astype(np.float32)
    indices = list(range(0, len(state), stride))
    num_frames = len(indices)

    return {
        "image": [resize_frame(cam0[i], image_size) for i in indices],
        "wrist_image": [resize_frame(cam1[i], image_size) for i in indices],
        "state": [state[i].tolist() for i in indices],
        "actions": [action[i].tolist() for i in indices],
        "prompt": [task] * num_frames,
        "episode_index": [episode_index] * num_frames,
        "frame_index": list(range(num_frames)),
        "index": list(range(global_offset, global_offset + num_frames)),
        "task_index": [task_index] * num_frames,
        "timestamp": [float(i / FPS) for i in indices],
    }


def materialize_for_stats(columns: dict[str, list], features: dict) -> dict:
    stats_data = {}
    for key, feature in features.items():
        if feature["dtype"] == "image":
            continue
        stats_data[key] = np.asarray(columns[key])
    return stats_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw pick-place PKLs into a local LeRobot dataset.")
    parser.add_argument("--input-dir", default="/nfs_global/S/yangrongzheng/pick_place_raw_data")
    parser.add_argument("--output-dir", default="/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--action-mode", choices=["joint_delta", "eef_delta", "joint_and_gripper_delta"], default="joint_and_gripper_delta")
    parser.add_argument("--use-next-state-action", action="store_true", help="Use next state as action target instead of delta. Mirrors newRL processed zarr semantics.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.pkl"))
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No pkl files found in {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)

    sample = None
    for candidate in files:
        try:
            sample = load_episode(candidate)
            validate_episode(sample, candidate)
            break
        except Exception:
            continue
    if sample is None:
        raise RuntimeError("No valid episode found to initialize dataset schema.")

    image_size = (args.image_height, args.image_width)
    state_dim = sample["episode_ur5e_pos_j"].shape[1] + sample["episode_ur5e_pos_eef"].shape[1] + sample["episode_inspire_hand_pos"].shape[1]
    action_dim = compute_action(sample, args.action_mode, use_next_state_action=args.use_next_state_action).shape[1]

    print(f"Preparing LeRobot dataset from {len(files)} candidate episodes")
    print(f"Output dir: {output_dir}")
    print(f"Image size: {args.image_height}x{args.image_width}")
    print(f"Stride: {args.stride}")
    print(f"Action mode: {args.action_mode}")
    print(f"Use next-state action: {args.use_next_state_action}")
    print(f"Stop on error: {args.stop_on_error}")

    meta = LeRobotDatasetMetadata.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=FPS,
        robot_type="ur5e_inspire_hand",
        features=build_features(state_dim, action_dim, image_size, include_task=True),
        use_videos=False,
    )

    summary = []
    skipped = []
    global_offset = 0
    all_episode_stats = []
    success_episode_index = 0
    progress = tqdm(files, desc="Converting episodes", unit="episode")
    for file_path in progress:
        progress.set_postfix_str(file_path.name)
        try:
            raw = load_episode(file_path)
            validate_episode(raw, file_path)
            task = make_task_text(parse_name(file_path.stem))
            if meta.get_task_index(task) is None:
                meta.add_task(task)
            task_index = meta.get_task_index(task)
            columns = build_columns(raw, args.action_mode, success_episode_index, task, task_index, global_offset, args.stride, image_size, use_next_state_action=args.use_next_state_action)

            hf_dataset = Dataset.from_dict(columns, features=get_hf_features_from_features(meta.features))
            data_path = output_dir / meta.get_data_file_path(success_episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            hf_dataset.to_parquet(str(data_path))

            stats_features = {key: value for key, value in meta.features.items() if value["dtype"] != "image"}
            episode_stats = compute_episode_stats(materialize_for_stats(columns, meta.features), stats_features)
            all_episode_stats.append(episode_stats)
            episode_length = len(columns["index"])

            meta.info["total_episodes"] += 1
            meta.info["total_frames"] += episode_length
            meta.info["total_chunks"] = max(meta.info["total_chunks"], meta.get_episode_chunk(success_episode_index) + 1)
            meta.info["splits"] = {"train": f"0:{meta.info['total_episodes']}"}

            write_episode({"episode_index": success_episode_index, "tasks": [task], "length": episode_length}, output_dir)
            write_episode_stats(success_episode_index, episode_stats, output_dir)
            write_info(meta.info, output_dir)

            global_offset += episode_length
            summary.append({"episode_index": success_episode_index, "source_file": file_path.name, "prompt": task, "length": episode_length})
            success_episode_index += 1
            print(f"converted episode {success_episode_index}: {file_path.name} -> {episode_length} frames")
        except Exception as exc:
            error_record = {
                "source_file": file_path.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            skipped.append(error_record)
            print(f"skipped bad episode: {file_path.name} ({type(exc).__name__}: {exc})", file=sys.stderr)
            if args.stop_on_error:
                raise

    meta.info["total_tasks"] = len(meta.tasks)
    if all_episode_stats:
        meta.stats = aggregate_stats(all_episode_stats)
    write_info(meta.info, output_dir)

    with (output_dir / "conversion_summary.json").open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with (output_dir / "conversion_skipped.json").open("w") as file:
        json.dump(skipped, file, indent=2, ensure_ascii=False)

    print(f"done: {len(summary)} episodes written to {output_dir}")
    print(f"skipped: {len(skipped)} bad episodes")


if __name__ == "__main__":
    main()
