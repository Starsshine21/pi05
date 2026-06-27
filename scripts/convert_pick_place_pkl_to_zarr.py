#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr


FPS = 10
DEFAULT_INPUT_DIR = "/nfs_global/S/yangrongzheng/pick_place_raw_data"
DEFAULT_OUTPUT_DIR = "/nfs_global/S/yangrongzheng/pi05/data/zarr_pick_place_eef_delta.zarr"
REQUIRED_KEYS = [
    "episode_ur5e_pos_j",
    "episode_ur5e_pos_eef",
    "episode_inspire_hand_pos",
    "episode_l515_color",
    "episode_orbbec_femto_bolt_color",
]


def load_episode(path: Path) -> dict:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path.name}: {missing}")
    return data


def validate_episode(data: dict, path: Path) -> int:
    joints = np.asarray(data["episode_ur5e_pos_j"])
    eef = np.asarray(data["episode_ur5e_pos_eef"])
    hand = np.asarray(data["episode_inspire_hand_pos"])
    cam0 = np.asarray(data["episode_l515_color"])
    cam1 = np.asarray(data["episode_orbbec_femto_bolt_color"])
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"Invalid joint shape in {path.name}: {joints.shape}")
    if eef.ndim != 2 or eef.shape[1] != 6:
        raise ValueError(f"Invalid eef shape in {path.name}: {eef.shape}")
    if hand.ndim != 2 or hand.shape[1] != 6:
        raise ValueError(f"Invalid hand shape in {path.name}: {hand.shape}")
    if cam0.ndim != 4 or cam0.shape[-1] != 3:
        raise ValueError(f"Invalid L515 image shape in {path.name}: {cam0.shape}")
    if cam1.ndim != 4 or cam1.shape[-1] != 3:
        raise ValueError(f"Invalid Orbbec image shape in {path.name}: {cam1.shape}")
    lengths = [len(joints), len(eef), len(hand), len(cam0), len(cam1)]
    if min(lengths) <= 1 or len(set(lengths)) != 1:
        raise ValueError(f"Mismatched trajectory lengths in {path.name}: {lengths}")
    return lengths[0]


def resize_nhwc(images: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in images], axis=0)


def compute_actions(eef: np.ndarray, hand: np.ndarray, use_next_state: bool) -> np.ndarray:
    base = np.concatenate([eef, hand], axis=1).astype(np.float32)
    action = np.zeros_like(base, dtype=np.float32)
    if use_next_state:
        action[:-1] = base[1:]
        action[-1] = base[-1]
    else:
        action[:-1] = base[1:] - base[:-1]
        action[-1] = action[-2] if len(action) > 1 else 0
    return action


def build_task(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 5:
        return stem.replace("_", " ")
    obj_name = parts[3].replace("-", " ")
    surface = parts[2].replace("-", " ")
    target = "_".join(parts[4:]).replace("_", " ").replace("-", " ")
    return f"place the {obj_name} at the {target} position on the {surface}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pick-place PKL episodes into zarr with eef-delta actions.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-next-state-action", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    files = sorted(input_dir.glob("*.pkl"))
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No PKL files found in {input_dir}")

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    episode_ends = []
    all_state = []
    all_action = []
    all_head = []
    all_wrist = []
    summary = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "action_semantics": "next_eef_hand" if args.use_next_state_action else "delta_eef_hand",
        "state_semantics": "concat(joints,eef,hand)",
        "image_mapping": {"head_camera": "episode_l515_color", "wrist_camera": "episode_orbbec_femto_bolt_color"},
        "fps": FPS,
        "episodes": [],
    }

    total = 0
    for idx, file_path in enumerate(files):
        raw = load_episode(file_path)
        validate_episode(raw, file_path)
        joints = np.asarray(raw["episode_ur5e_pos_j"], dtype=np.float32)
        eef = np.asarray(raw["episode_ur5e_pos_eef"], dtype=np.float32)
        hand = np.asarray(raw["episode_inspire_hand_pos"], dtype=np.float32)
        l515 = resize_nhwc(np.asarray(raw["episode_l515_color"], dtype=np.uint8), args.image_height, args.image_width)
        orbbec = resize_nhwc(np.asarray(raw["episode_orbbec_femto_bolt_color"], dtype=np.uint8), args.image_height, args.image_width)
        state = np.concatenate([joints, eef, hand], axis=1).astype(np.float32)
        action = compute_actions(eef, hand, args.use_next_state_action)
        prompt = build_task(file_path.stem)

        all_state.append(state)
        all_action.append(action)
        all_head.append(np.moveaxis(l515, -1, 1))
        all_wrist.append(np.moveaxis(orbbec, -1, 1))
        total += len(state)
        episode_ends.append(total)
        summary["episodes"].append({"episode_index": idx, "file": file_path.name, "length": len(state), "prompt": prompt})

    state = np.concatenate(all_state, axis=0)
    action = np.concatenate(all_action, axis=0)
    head = np.concatenate(all_head, axis=0)
    wrist = np.concatenate(all_wrist, axis=0)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.open_group(str(output_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    data_group.create_dataset("state", data=state, chunks=(256, state.shape[1]), dtype="f4", compressor=compressor)
    data_group.create_dataset("action", data=action, chunks=(256, action.shape[1]), dtype="f4", compressor=compressor)
    data_group.create_dataset("head_camera", data=head, chunks=(32, *head.shape[1:]), dtype="u1", compressor=compressor)
    data_group.create_dataset("wrist_camera", data=wrist, chunks=(32, *wrist.shape[1:]), dtype="u1", compressor=compressor)
    meta_group.create_dataset("episode_ends", data=episode_ends_arr, dtype="i8", compressor=compressor)

    stats = {
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": state.std(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
    }
    summary["stats"] = stats
    summary["num_episodes"] = len(files)
    summary["num_frames"] = int(total)
    (output_path / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "num_episodes": len(files), "num_frames": int(total), "action_semantics": summary["action_semantics"]}, indent=2))


if __name__ == "__main__":
    main()
