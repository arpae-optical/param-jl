#!/usr/bin/env python3


from __future__ import annotations

from math import floor
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, TypedDict

import numpy as np
import torch
from torch import Tensor

Stage = Literal["train", "val", "test"]

# TODO replace with scanl
def split(n: int, splits: Optional[Mapping[Stage, float]] = None) -> Dict[Stage, range]:
    """
    n: length of dataset
    splits: map where values should sum to 1 like in `{"train": 0.8, "val": 0.1, "test": 0.1}`
    """
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


def rmse(pred: Tensor, target: Tensor, epsilon=1e-8):
    """Root mean squared error.

    Epsilon is to avoid NaN gradients. See https://discuss.pytorch.org/t/rmse-loss-function/16540/3.
    """
    return (torch.nn.functional.mse_loss(pred, target) + epsilon).sqrt()


def step_tensor():
    """Returns tensor of wavelengths, sorted high to low."""
    # index at 0 because each row has the same info.
    wavelens = torch.load(Path("wavelength.pt"))[0]
    out = torch.zeros(len(wavelens), len(wavelens))
    for r, _ in enumerate(out):
        out[r, : r + 1] = 1.0
    return out


class Config(TypedDict):
    forward_lr: float
    backward_lr: float
    forward_num_epochs: int
    backward_num_epochs: int
    forward_batch_size: int
    backward_batch_size: int
    use_cache: bool
    num_wavelens: int
    use_forward:bool
    load_forward_checkpoint:bool
    load_backward_checkpoint:bool
