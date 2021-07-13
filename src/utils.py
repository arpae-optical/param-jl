#!/usr/bin/env python3


from __future__ import annotations
from torch import Tensor

from math import floor
from typing import (
    Dict,
    Literal,
    Mapping,
    Optional,
)

import torch

Stage = Literal["train", "val", "test"]

# TODO replace with scanl
def split(n: int, splits: Optional[Mapping[Stage, float]] = None) -> Dict[Stage, range]:
    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    return {
        "train": range(0, floor(n * splits["train"])),
        "val": range(
            floor(n * splits["train"]),
            floor(n * splits["train"]) + floor(n * splits["val"]),
        ),
        "test": range(floor(n * splits["train"]) + floor(n * splits["val"]), n),
    }


def rmse(pred: Tensor, target: Tensor):
    return torch.nn.functional.mse_loss(pred, target).sqrt()
