#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import pymongo
import pytorch_lightning as pl
import sklearn
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib import tenumerate

import utils
from utils import Config, Stage, rmse, split

LaserParams, Emiss = FloatTensor, FloatTensor


def get_data(
    use_cache: bool = True, num_wavelens: int = 300
) -> Tuple[LaserParams, Emiss, LongTensor]:
    """Data is sorted in ascending order of wavelength."""
    if all(
        [
            use_cache,
            Path("/data-new/alok/laser/data.pt").exists(),
        ]
    ):
        data = torch.load(Path("/data-new/alok/laser/data.pt"))
        norm_laser_params, interp_emissivities, uids = (
            data["normalized_laser_params"],
            data["interpolated_emissivity"],
            data["uids"],
        )

        # XXX check length to avoid bugs.
        if interp_emissivities.shape[-1] == num_wavelens:
            return norm_laser_params, interp_emissivities, uids

    client = pymongo.MongoClient(
        "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
    )
    db = client.propopt.laser_samples2
    laser_params, emissivity, wavelength = [], [], []
    interp_emissivities, interp_wavelengths = [], []
    uids = []
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
    for uid, entry in tenumerate(db.find()):
        # TODO: ensure that this is sorted by wavelength
        # TODO log transform?
        emiss_plot: List[float] = [
            e
            for ex in entry["emissivity_spectrum"]
            if ((e := ex["normal_emissivity"]) != 1.0 and ex["wavelength_micron"] < 12)
        ]
        wavelength_plot: List[float] = [
            ex["wavelength_micron"]
            for ex in entry["emissivity_spectrum"]
            if (ex["normal_emissivity"] != 1.0 and ex["wavelength_micron"] < 12)
        ]
        # Reverse to sort in ascending rather than descending order.
        emiss_plot.reverse()
        wavelength_plot.reverse()
        # interpolated columns
        interp_wavelen = np.linspace(
            min(wavelength_plot), max(wavelength_plot), num=num_wavelens
        )
        interp_emiss = interp1d(wavelength_plot, emiss_plot)(interp_wavelen)

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
        uids.append(uid)
        laser_params.append(params)
        emissivity.append(emiss_plot)
        wavelength.append(wavelength_plot)
        interp_emissivities.append(interp_emiss)
        interp_wavelengths.append(interp_wavelen)

    # normalize laser parameters
    laser_params = FloatTensor(laser_params)
    emissivity = FloatTensor(emissivity)
    wavelength = FloatTensor(wavelength)
    interp_emissivities = FloatTensor(interp_emissivities)
    interp_wavelengths = FloatTensor(interp_wavelengths)
    uids = LongTensor(uids)

    print(f"{len(laser_params)=}")
    print(f"{len(emissivity)=}")
    print(f"{laser_params.min(0)=}")
    print(f"{laser_params.max(0)=}")
    print(f"{emissivity.min()=}")
    print(f"{emissivity.max()=}")

    # Save unnormalized data for convenience later.

    norm_laser_params = laser_params / laser_params.max(0).values
    torch.save(
        {
            "wavelength": wavelength,
            "laser_params": laser_params,
            "emissivity": emissivity,
            "uids": uids,
            "interpolated_emissivity": interp_emissivities,
            "interpolated_wavelength": interp_wavelengths,
            "normalized_laser_params": norm_laser_params,
        },
        Path("/data-new/alok/laser/data.pt"),
    )

    return norm_laser_params, interp_emissivities, uids


class ForwardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["forward_batch_size"]

    def setup(self, stage: Optional[str]) -> None:

        laser_params, emiss, uids = get_data(
            use_cache=self.config["use_cache"], num_wavelens=self.config["num_wavelens"]
        )
        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                laser_params[splits[s].start : splits[s].stop],
                emiss[splits[s].start : splits[s].stop],
                uids[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, "/data/alok/laser/forward_train_true.pt")
        torch.save(self.val, "/data/alok/laser/forward_val_true.pt")
        torch.save(self.test, "/data/alok/laser/forward_test_true.pt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )


class BackwardDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["backward_batch_size"]

    def setup(self, stage: Optional[str]) -> None:

        # XXX we can safely use cache since ForwardDataModule is created first
        laser_params, emiss, uids = get_data(
            use_cache=True, num_wavelens=self.config["num_wavelens"]
        )

        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                emiss[splits[s].start : splits[s].stop],
                laser_params[splits[s].start : splits[s].stop],
                uids[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]

        torch.save(self.train, "/data/alok/laser/backward_train_true.pt")
        torch.save(self.val, "/data/alok/laser/backward_val_true.pt")
        torch.save(self.test, "/data/alok/laser/backward_test_true.pt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )


class StepTestDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str]) -> None:
        self.test = TensorDataset(utils.step_tensor())

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=1_000,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )


class TPVData(TypedDict):
    raw_emiss: Tensor
    raw_wavelen: Tensor
    interp_emiss: Tensor
    interp_wavelen: Tensor
    power: float
    speed: float
    spacing: float


def parse_all() -> None:
    """For converting Minok's raw data."""

    def parse_entry(filename: Path) -> TPVData:
        pattern = re.compile(
            r"Power_(\d)_(\d)_W_Speed_(.+)_mm_s_Spacing_(.+)_um\s*(\d+)\s*\.txt"
        )
        m = pattern.match(filename.name)
        if m is None:
            raise ValueError(f"Could not parse filename {filename}")
        power, x_speed, y_spacing, rand_id = (
            int(m[1]) + int(m[2]) * 10**-1,
            float(m[3]),
            float(m[4]),
            int(m[5]),
        )
        raw_data = pd.read_csv(
            filename, header=None, names=["wavelens", "emisses"], delim_whitespace=True
        )
        # Reverse order to match other data. The `.copy()` is to avoid a "negative stride" conversion error.
        all_wavelens = raw_data.wavelens[::-1].copy()
        emisses = raw_data.emisses[::-1].copy()
        # clip to same as rest of data
        MAX_WAVELEN = 12
        wavelens = all_wavelens.loc[all_wavelens < MAX_WAVELEN].values
        emisses = emisses.loc[all_wavelens < MAX_WAVELEN].values
        # wavelens = all_wavelens.values
        # emisses = emisses.values

        interp_wavelen = np.linspace(min(wavelens), max(wavelens), num=800)
        interp_emiss = interp1d(wavelens, emisses)(interp_wavelen)

        return {
            "raw_emiss": torch.as_tensor(emisses),
            "interp_emiss": torch.as_tensor(interp_emiss),
            "interp_wavelen": torch.as_tensor(interp_wavelen),
            "raw_wavelen": torch.as_tensor(wavelens),
            "power": power,
            "speed": x_speed,
            "spacing": y_spacing,
        }
        # client = pymongo.MongoClient(
        #     "mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt"
        # )
        # db = client.propopt.laser_samples3
        # db.insert(entry)

    out_tpv = [
        parse_entry(p)
        for p in Path(
            "/data-new/alok/laser/minok_ml_data2/Validation_2nd/Full_New_TPV"
        ).rglob("*.txt")
    ]
    out_preds = [
        parse_entry(p)
        for p in Path(
            "/data-new/alok/laser/minok_ml_data2/Validation_2nd/Direct_Predictions"
        ).rglob("*.txt")
    ]
    torch.save(out_tpv, Path("/data-new/alok/laser/minok_tpv_data.pt"))
    torch.save(out_preds, Path("/data-new/alok/laser/minok_laser_preds.pt"))


if __name__ == "__main__":
    parse_all()
