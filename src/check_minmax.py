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
)

import pymongo


def minmax():
    client = pymongo.MongoClient(
        "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
    )
    db = client.propopt.laser_samples2
    speed, spacing = [], []
    for entry in db.find():
        emiss_plot: List[float] = [
            e
            for ex in entry["emissivity_spectrum"]
            if ((e := ex["normal_emissivity"]) != 1.0 and ex["wavelength_micron"] < 12)
        ]
        if len(emiss_plot) != (_MANUALLY_COUNTED_LENGTH := 821) or any(
            not (0 <= x <= 1) for x in emiss_plot
        ):
            continue
        if entry["laser_power_W"] > 1.3:
            print(entry["laser_power_W"])
        speed.append(entry["laser_scanning_speed_x_dir_mm_per_s"])
        spacing.append(entry["laser_scanning_line_spacing_y_dir_micron"]),

    print(f"{max(speed), min(speed), max(spacing), min(spacing)=}")
