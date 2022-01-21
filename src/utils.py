#!/usr/bin/env python3


from __future__ import annotations

from math import floor
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

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


def rmse(pred: Tensor, target: Tensor):
    """Root mean squared error."""
    return torch.nn.functional.mse_loss(pred, target).sqrt()


def step_tensor():
    """Returns tensor of wavelengths, sorted high to low."""
    # index at 0 because each row has the same info.
    wavelens = torch.load(Path("wavelength.pt"))[0]
    out = torch.zeros(len(wavelens), len(wavelens))
    for r, row in enumerate(out):
        out[r, : r + 1] = 1.0
    return out
