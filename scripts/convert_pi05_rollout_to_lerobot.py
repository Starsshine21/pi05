#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_hf_features_from_features, write_episode, write_episode_stats, write_info

FPS = 10
REPO_ID = 'local/pi05-pickplace-il-online'


def build_features(state_dim: int, action_dim: int):
    return {
        'image': {'dtype': 'image', 'shape': [224, 224, 3], 'names': ['height', 'width', 'channel']},
        'wrist_image': {'dtype': 'image', 'shape': [224, 224, 3], 'names': ['height', 'width', 'channel']},
        'state': {'dtype': 'float32', 'shape': [state_dim], 'names': None},
        'actions': {'dtype': 'float32', 'shape': [action_dim], 'names': None},
        'prompt': {'dtype': 'string', 'shape': [1], 'names': None},
        'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
    }


def materialize_for_stats(columns: dict[str, list], features: dict) -> dict:
    stats_data = {}
    for key, feature in features.items():
        if feature['dtype'] == 'image':
            continue
        stats_data[key] = np.asarray(columns[key])
    return stats_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert collected PI05 real rollouts to LeRobot format.')
    parser.add_argument('--input-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_real_collect')
    parser.add_argument('--output-dir', default='/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online')
    parser.add_argument('--repo-id', default=REPO_ID)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / 'raw_rollouts'
    ep_dirs = sorted(p for p in input_dir.glob('episode_*') if p.is_dir())
    if not ep_dirs:
        raise FileNotFoundError(f'No rollout episodes found in {input_dir}')

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'Output exists: {output_dir}. Use --overwrite.')
        shutil.rmtree(output_dir)

    first = np.load(ep_dirs[0] / 'frames.npz')
    state_dim = int(first['state'].shape[1])
    action_dim = int(first['actions'].shape[1])
    meta = LeRobotDatasetMetadata.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=FPS,
        robot_type='ur5e_inspire_hand',
        features=build_features(state_dim, action_dim),
        use_videos=False,
    )

    global_offset = 0
    all_episode_stats = []
    for success_episode_index, ep_dir in enumerate(ep_dirs):
        frames = np.load(ep_dir / 'frames.npz')
        ep_meta = json.loads((ep_dir / 'meta.json').read_text())
        task = ep_meta['prompt']
        if meta.get_task_index(task) is None:
            meta.add_task(task)
        task_index = meta.get_task_index(task)
        n = int(frames['state'].shape[0])
        columns = {
            'image': [img for img in frames['image']],
            'wrist_image': [img for img in frames['wrist_image']],
            'state': [row.tolist() for row in frames['state']],
            'actions': [row.tolist() for row in frames['actions']],
            'prompt': [task] * n,
            'episode_index': [success_episode_index] * n,
            'frame_index': list(range(n)),
            'index': list(range(global_offset, global_offset + n)),
            'task_index': [task_index] * n,
            'timestamp': [float(x) for x in frames['timestamp']],
        }
        hf_dataset = Dataset.from_dict(columns, features=get_hf_features_from_features(meta.features))
        data_path = output_dir / meta.get_data_file_path(success_episode_index)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        hf_dataset.to_parquet(str(data_path))

        stats_features = {key: value for key, value in meta.features.items() if value['dtype'] != 'image'}
        episode_stats = compute_episode_stats(materialize_for_stats(columns, meta.features), stats_features)
        all_episode_stats.append(episode_stats)
        write_episode(meta.episodes_path, success_episode_index, task, n)
        write_episode_stats(meta.episodes_stats_path, success_episode_index, episode_stats)
        global_offset += n

    if all_episode_stats:
        aggregated = aggregate_stats(all_episode_stats)
        write_info(meta.info_path, meta.features, meta.total_episodes, meta.total_frames, meta.total_tasks, meta.total_videos, meta.total_chunks, meta.chunks_size, meta.fps, meta.splits, meta.data_path, meta.video_path)
        stats_path = output_dir / 'meta' / 'stats.json'
        stats_path.write_text(json.dumps({k: {kk: np.asarray(vv).tolist() for kk, vv in vals.items()} for k, vals in aggregated.items()}, indent=2))
    print(output_dir)


if __name__ == '__main__':
    main()
