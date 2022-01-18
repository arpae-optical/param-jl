#!/usr/bin/env python3


from __future__ import annotations

from math import floor
from typing import Dict, Literal, Mapping, Optional
from pathlib import Path
import torch
from torch import Tensor
import numpy as np

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

def step_at_n(n: float = 3.5, max: float = 12):
    """Returns tuple of x and y which corresponds to a function that outputs 0 at index < n, and 1 at index > n.

    n: wavelength to step down at
    max: max wavelength
    
    
    """
    
    wavelength = torch.load(Path("wavelength.pt"))
    low = 0
    high = 0
    for i, wl in enumerate(wavelength[0]):
        if wl < n:
            low = i
        if wl < max:
            high = i

    x = wavelength[0][low:high]
    y = [1 for i in range(low)] + [0 for i in range(high-low)]
    return (x, y)