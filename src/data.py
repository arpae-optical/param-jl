#!/usr/bin/env python3

from __future__ import annotations

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
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import pymongo
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm, trange

LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


def get_data(use_cache: bool = True) -> Tuple[LaserParams, Emiss]:
    if all(
        [
            use_cache,
            Path("emissivity.pt").exists(),
            Path("laser_params.pt").exists(),
        ]
    ):
        laser_params, emissivity = torch.load(Path("laser_params.pt")), torch.load(
            Path("emissivity.pt")
        )
    else:
        client = pymongo.MongoClient(
            "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
        )
        db = client.propopt.laser_samples
        laser_params, emissivity = [], []

        for entry in tqdm(db.find()):
            # XXX: chop off the top emissivity since it's always 1 and I think that's a bug. The `[1:]` does that
            emiss_plot: List[float] = [
                ex["normal_emissivity"] for ex in entry["emissivity_spectrum"][1:]
            ]
            # drop all problematic emissivity (only 3% of data dropped)
            # XXX The `935 - 1` is to account for the chopping off above.
            if len(emiss_plot) != (935 - 1) or any(
                not (0 <= x <= 1) for x in emiss_plot
            ):
                continue

            laser_params.append(
                [
                    entry["laser_scanning_speed_x_dir_mm_per_s"],
                    entry["laser_scanning_line_spacing_y_dir_micron"],
                    # XXX laser_rep_rate and wavelength_nm are all the same
                    # float(entry["laser_repetition_rate_kHz"]),
                    # float(entry["laser_wavelength_nm"]),
                    float(entry["laser_power_W"]),
                ]
            )
            emissivity.append(emiss_plot)

        # normalize laser parameters
        laser_params = torch.FloatTensor(laser_params)
        emissivity = torch.FloatTensor(emissivity)

        print(f"{len(laser_params)=}")
        print(f"{len(emissivity)=}")
        print(f"{laser_params.min()=}")
        print(f"{laser_params.max()=}")
        print(f"{emissivity.min()=}")
        print(f"{emissivity.max()=}")

        laser_params = laser_params / laser_params.max(0).values

        torch.save(laser_params, Path("laser_params.pt"))
        torch.save(emissivity, Path("emissivity.pt"))

    return laser_params, emissivity
