import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.libero_policy import LiberoInputs


@dataclasses.dataclass(frozen=True)
class PickPlaceInputs(LiberoInputs):
    pass


@dataclasses.dataclass(frozen=True)
class PickPlaceOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :12])}
