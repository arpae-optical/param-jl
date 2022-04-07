#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pymongo
import pytorch_lightning as pl
import sklearn
import torch
import torch.nn.functional as F
from pl_bolts.datamodules import SklearnDataset
from scipy.interpolate import interp1d
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from tqdm import tqdm
from tqdm.contrib import tenumerate
from typeguard import typechecked

import utils
from utils import Config, Stage, rmse, split

patch_typeguard()


def create_dataset(
    save_path: Path = Path("/data/alok/laser/dataset.pkl"),
    use_cache: bool = True,
    num_wavelens: int = 300,
):
    def filter(dataset: pd.DataFrame) -> pd.DataFrame:
        return DataFrameMapper(
            [(["wattage"], OrdinalEncoder(), {"alias": "encoded_wattage"})],
            default=None,  # passes other columns through
            df_out=True,
        ).fit_transform(dataset)

    if use_cache and save_path.exists():
        return pd.read_pickle(save_path)

    db = pymongo.MongoClient(
        # "mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt"
        host="mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
    ).propopt.laser_samples2

    # row names match dataframe names
    rows = []
    for uid, entry in tenumerate(db.find()):

        # row names match dataframe names
        row = {
            # Reverse so wavelen/emiss are sorted ascending.
            "emissivity": list(
                reversed(
                    [
                        e["normal_emissivity"]
                        for e in entry["emissivity_spectrum"]
                        if 0.0 < e["normal_emissivity"] < 1.0
                        and e["wavelength_micron"] < 12.0
                    ]
                )
            ),
            "wavelength": list(
                reversed(
                    [
                        e["wavelength_micron"]
                        for e in entry["emissivity_spectrum"]
                        if e["wavelength_micron"] < 12.0
                    ]
                )
            ),
            "uid": uid,
            "wattage": entry["laser_power_W"],
            "speed": entry["laser_scanning_speed_x_dir_mm_per_s"],
            "spacing": entry["laser_scanning_line_spacing_y_dir_micron"],
        }

        # interpolated columns
        row["interpolated_wavelength"] = np.linspace(
            min(row["wavelength"]), max(row["wavelength"]), num=num_wavelens
        )
        row["interpolated_emissivity"] = interp1d(row["wavelength"], row["emissivity"])(
            row["interpolated_wavelength"]
        )

        # TODO should we bring this back?
        if row["wattage"] > 1.3 or len(row["emissivity"]) != 821:
            continue
        else:
            rows.append(row)
    dataset = pd.DataFrame(rows)
    dataset = filter(dataset)
    dataset.to_pickle(save_path)

    return dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["batch_size"]

    def setup(self, stage: Optional[str]) -> None:
        T = torch.as_tensor
        df = create_dataset(use_cache=self.config["use_cache"])
        emiss: TT["b", "num_wavelens"] = torch.stack(
            [torch.as_tensor(row) for row in (df["emissivity"])]
        )
        laser_params: TT["b", 3] = torch.stack(
            [
                torch.cat([T(row.spacing), T(row.speed), T(row.encoded_wattage)])
                for row in df.iterrows()
            ]
        )
        uids: TT["b"] = T([row.uid for row in df.iterrows()])

        splits = split(len(df))

        self.train, self.val, self.test = [
            TensorDataset(
                emiss[splits[s].start : splits[s].stop],
                laser_params[splits[s].start : splits[s].stop],
                uids[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]

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


def parse_all() -> None:
    """For converting Minok's raw data."""

    def parse_entry(filename: Path) -> None:
        pattern = re.compile(r"Power_(\d)_(\d)_W_Speed_(\d+)_mm_s_Spacing_(\d+)_um.txt")
        m = pattern.match(filename.name)
        if m is None:
            return
        power1, power2, x_speed, y_spacing = m[1], m[2], m[3], m[4]
        x_speed = float(x_speed)
        y_spacing = float(y_spacing)
        power = int(power1) + int(power2) * 10**-1
        raw_data = pd.read_csv(
            filename, header=None, names=["wavelens", "emisses"], delim_whitespace=True
        )
        wavelens = raw_data.wavelens
        emisses = raw_data.emisses
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

    for p in Path("/home/alok/minok_ml_data").rglob("*.txt"):
        parse_entry(p)


if __name__ == "__main__":
    df = create_dataset(use_cache=False)
