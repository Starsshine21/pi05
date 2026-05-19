"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _select_episodes_for_max_frames(dataset_meta, max_frames: int | None) -> list[int] | None:
    if max_frames is None:
        return None

    selected = []
    total_frames = 0
    for episode_index in sorted(dataset_meta.episodes):
        selected.append(episode_index)
        total_frames += int(dataset_meta.episodes[episode_index]["length"])
        if total_frames >= max_frames:
            break

    print(
        "create_torch_dataloader: "
        f"selected_episodes={selected}, selected_frames={total_frames}, requested_max_frames={max_frames}",
        flush=True,
    )
    return selected


def _create_torch_dataset_for_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None,
) -> _data_loader.Dataset:
    if max_frames is None or data_config.repo_id == "fake":
        return _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")

    dataset_meta = _data_loader.lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    episodes = _select_episodes_for_max_frames(dataset_meta, max_frames)
    dataset = _data_loader.lerobot_dataset.LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    print("create_torch_dataloader: loading LeRobot dataset metadata/cache...", flush=True)
    dataset = _create_torch_dataset_for_stats(data_config, action_horizon, model_config, max_frames)
    print(f"create_torch_dataloader: dataset_len={len(dataset)}", flush=True)
    print("create_torch_dataloader: attaching repack/data transforms...", flush=True)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max(1, int(np.ceil(max_frames / batch_size)))
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    print(
        "create_torch_dataloader: "
        f"batch_size={batch_size}, num_workers={num_workers}, num_batches={num_batches}, shuffle={shuffle}",
        flush=True,
    )
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    print("create_rlds_dataloader: loading RLDS dataset...", flush=True)
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max(1, int(np.ceil(max_frames / batch_size)))
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    num_workers: int | None = None,
    assets_base_dir: str | None = None,
):
    print(f"loading config: {config_name}", flush=True)
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    effective_num_workers = config.num_workers if num_workers is None else num_workers

    print(f"config_name={config_name}", flush=True)
    print(f"repo_id={data_config.repo_id}", flush=True)
    print(f"batch_size={config.batch_size}", flush=True)
    print(f"num_workers={effective_num_workers}", flush=True)
    print(f"max_frames={max_frames}", flush=True)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, effective_num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    print(f"num_batches={num_batches}", flush=True)
    total_frames = 0

    print("starting stats loop", flush=True)
    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"), start=1):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
        batch_frames = int(np.asarray(batch["state"]).shape[0])
        total_frames += batch_frames
        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_batches:
            print(
                f"progress: batch {batch_idx}/{num_batches}, "
                f"batch_frames={batch_frames}, total_frames={total_frames}",
                flush=True,
            )

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}", flush=True)
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
