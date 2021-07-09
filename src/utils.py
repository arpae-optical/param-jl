#!/usr/bin/env python3


from __future__ import annotations
from torch import Tensor

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from math import floor
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
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

def rmse(pred:Tensor,target:Tensor):
    return torch.nn.functional.mse_loss(pred,target).sqrt()
