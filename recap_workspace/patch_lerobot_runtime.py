import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def patch_lerobot_query_hf_dataset() -> None:
    if getattr(LeRobotDataset, '_recap_query_patch_applied', False):
        return

    def _query_hf_dataset_compat(self, query_indices):
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            values = self.hf_dataset.select(q_idx)[key]
            if hasattr(values, 'to_pylist'):
                values = values.to_pylist()
            else:
                values = list(values)
            if values and isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    LeRobotDataset._query_hf_dataset = _query_hf_dataset_compat
    LeRobotDataset._recap_query_patch_applied = True
