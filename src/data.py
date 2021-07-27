#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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
            # XXX: chop off the top emissivity since it's always 1 and I think that's a bug. The `[1:]` does that
            # TODO: ensure that this is sorted by wavelength
            # TODO log transform?
            emiss_plot: List[float] = [
                e
                for ex in entry["emissivity_spectrum"]
                if (e := ex["normal_emissivity"]) != 1.0
            ]
            # drop all problematic emissivity (only 3% of data dropped)
            # XXX The `935 - 1` is to account for the chopping off above.
            if len(emiss_plot) != (935 - 1) or any(
                not (0 <= x <= 1) for x in emiss_plot
            ):
                continue
            if entry["laser_power_W"] > 1.3:
                print(entry["laser_power_W"])
            params = [
                entry["laser_scanning_speed_x_dir_mm_per_s"],
                entry["laser_scanning_line_spacing_y_dir_micron"],
                # TODO these should be computed by the model doing actual averaging, not direct prediction
                entry["emissivity_averaged_over_frequency"],
                entry["emissivity_averaged_over_wavelength"],
                # XXX laser_rep_rate and wavelength_nm are all the same
                # float(entry["laser_repetition_rate_kHz"]),
                # float(entry["laser_wavelength_nm"]),
                *F.one_hot(
                    torch.tensor(wattage_idxs[entry["laser_power_W"]]),
                    num_classes=len(wattage_idxs),
                ),
            ]
            laser_params.append(params)
            emissivity.append(emiss_plot)

        # normalize laser parameters
        laser_params = torch.FloatTensor(laser_params)
        emissivity = torch.FloatTensor(emissivity)

        # break any correlations in data
        laser_params, emissivity = sklearn.utils.shuffle(laser_params, emissivity)

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
