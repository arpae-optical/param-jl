#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pymongo
import sklearn
import torch
import torch.nn.functional as F
from tqdm import tqdm

LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


def get_data(use_cache: bool = True) -> Tuple[LaserParams, Emiss]:
    if all(
        [
            use_cache,
            Path("emissivity.pt").exists(),
            Path("laser_params.pt").exists(),
            Path("wavelength.pt").exists(),
        ]
    ):
        laser_params, emissivity, wavelength = (
            torch.load(Path("laser_params.pt")),
            torch.load(Path("emissivity.pt")),
            torch.load(Path("wavelength.pt")),
        )
    else:
        client = pymongo.MongoClient(
            "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
        )
        db = client.propopt.laser_samples2
        laser_params, emissivity, wavelength = [], [], []
        wattages = []
        # TODO: clean up and generalize when needed
        # the values are indexes for one hot vectorization
        wattage_idxs = {
            0.2: 0,
            0.3: 1,
            0.4: 2,
            0.5: 3,
            0.6: 4,
            0.7: 5,
            0.8: 6,
            0.9: 7,
            1.0: 8,
            1.1: 9,
            1.2: 10,
            1.3: 11,
            # these last 2 wattages are problematic since their
            # emissivities are different lengths
            # 1.4: 12,
            # 1.5: 13,
        }

        # TODO: relax this to all wattages, try discretizing them w/
        # softmax instead
        for entry in tqdm(db.find()):
            # TODO: ensure that this is sorted by wavelength
            # TODO log transform?
            emiss_plot: List[float] = [
                e
                for ex in entry["emissivity_spectrum"]
                if (
                    (e := ex["normal_emissivity"]) != 1.0
                    and ex["wavelength_micron"] < 12
                )
            ]
            wavelength_plot: List[float] = [
                ex["wavelength_micron"]
                for ex in entry["emissivity_spectrum"]
                if (ex["normal_emissivity"] != 1.0 and ex["wavelength_micron"] < 12)
            ]
            # drop all problematic emissivity (only 3% of data dropped)

            if len(emiss_plot) != (_MANUALLY_COUNTED_LENGTH := 821) or any(
                not (0 <= x <= 1) for x in emiss_plot
            ):
                continue
            if entry["laser_power_W"] > 1.3:
                print(entry["laser_power_W"])
            params = [
                entry["laser_scanning_speed_x_dir_mm_per_s"],
                entry["laser_scanning_line_spacing_y_dir_micron"],
                *F.one_hot(
                    torch.tensor(wattage_idxs[round(entry["laser_power_W"], 1)]),
                    num_classes=len(wattage_idxs),
                ),
            ]
            laser_params.append(params)
            emissivity.append(emiss_plot)
            wavelength.append(wavelength_plot)

        # normalize laser parameters
        laser_params = torch.FloatTensor(laser_params)
        emissivity = torch.FloatTensor(emissivity)
        wavelength = torch.FloatTensor(wavelength)

        # break any correlations in data
        laser_params, emissivity, wavelength = sklearn.utils.shuffle(
            laser_params, emissivity, wavelength
        )

        print(f"{len(laser_params)=}")
        print(f"{len(emissivity)=}")
        print(f"{laser_params.min(0)=}")
        print(f"{laser_params.max(0)=}")
        print(f"{emissivity.min()=}")
        print(f"{emissivity.max()=}")

        laser_params /= laser_params.max(0).values

        torch.save(laser_params, Path("laser_params.pt"))
        torch.save(emissivity, Path("emissivity.pt"))
        torch.save(wavelength, Path("wavelength.pt"))

    return laser_params, emissivity


def parse_entry(filename: os.PathLike) -> None:
    pattern = re.compile("Power_(\d)_(\d)_W_Speed_(\d+)_mm_s_Spacing_(\d+)_um.txt")
    m = pattern.match(filename.name)
    if m is None:
        return
    power1, power2, x_speed, y_spacing = m[1], m[2], m[3], m[4]
    x_speed = float(x_speed)
    y_spacing = float(y_spacing)
    power = int(power1) + int(power2) * 10 ** -1
    data = pd.read_csv(
        filename, header=None, names=["wavelens", "emisses"], delim_whitespace=True
    )
    wavelens = data.wavelens
    emisses = data.emisses
    # emissivity_averaged_over_wavelength = ...
    entry = {
        "laser_repetition_rate_kHz": 100,
        "laser_wavelength_nm": 1030,
        "laser_polarization": "p-pol",
        "laser_steering_equipment": "Galvano scanner",
        "laser_hardware_model": "s-Pulse (Amplitude)",
        "substrate_details": "SS foil with 0.5 mm thickness (GF90624180-20EA)",
        "laser_power_W": power,
        "laser_scanning_speed_x_dir_mm_per_s": x_speed,
        "laser_scanning_line_spacing_y_dir_micron": y_spacing,
        "substrate_material": "stainless_steel",
        "emissivity_spectrum": [
            {"wavelength_micron": w, "normal_emissivity": e}
            for w, e in zip(wavelens, emisses)
        ],
        "emissivity_averaged_over_frequency": sum(emisses) / len(emisses),
    }
    client = pymongo.MongoClient(
        "mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt"
    )
    db = client.propopt.laser_samples2
    db.insert(entry)


def parse_all():
    for p in Path("/home/alok/minok_ml_data").rglob("*.txt"):
        parse_entry(p)
