#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Literal, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import pymongo
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm, trange

from utils import Stage, split

parser = argparse.ArgumentParser()

parser.add_argument("--num-epochs", "-n", type=int, default=50_000)
parser.add_argument("--batch-size", "-b", type=int, default=256)
args = parser.parse_args()


LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


def get_data() -> Tuple[LaserParams, Emiss]:
    client = pymongo.MongoClient(
        "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
    )
    db = client.propopt.laser_samples
    laser_params, emissivity = [], []

    for entry in tqdm(db.find()):
        emiss_plot: List[float] = [
            ex["normal_emissivity"] for ex in entry["emissivity_spectrum"]
        ]
        # drop all problematic emiss (only 3% of data dropped)
        if len(emiss_plot) != 935 or any(not (0 <= x <= 1) for x in emiss_plot):
            continue

        laser_params.append(
            [
                entry["laser_scanning_speed_x_dir_mm_per_s"],
                entry["laser_scanning_line_spacing_y_dir_micron"],
                float(entry["laser_repetition_rate_kHz"]),
            ]
        )
        emissivity.append(emiss_plot)

    laser_params, emissivity = torch.FloatTensor(laser_params), torch.FloatTensor(
        emissivity
    )
    print(f'{len(laser_params)=}')
    print(f'{len(emissivity)=}')
    print(f'{laser_params.min()=}')
    print(f'{laser_params.max()=}')
    print(f'{emissivity.min()=}')
    print(f'{emissivity.max()=}')
    return laser_params, emissivity


class ForwardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = args.batch_size,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str]) -> None:

        input, output = get_data()
        splits = split(len(input))
        self.train = TensorDataset(
            input[splits["train"].start : splits["train"].stop],
            output[splits["train"].start : splits["train"].stop],
        )
        self.val = TensorDataset(
            input[splits["val"].start : splits["val"].stop],
            output[splits["val"].start : splits["val"].stop],
        )
        self.test = TensorDataset(
            input[splits["test"].start : splits["test"].stop],
            output[splits["test"].start : splits["test"].stop],
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )


Mode = Literal["forward", "backward"]


class ForwardModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 935),
            nn.Sigmoid(),
        )
        # TODO how to reverse the *data* in the Linear layers easily? transpose?

        # TODO add mode arg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="train")

    def validation_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="test")

    def _step(self, batch, batch_nb, stage: Stage):
        x, y = batch
        y_pred = self(x)
        print(x)
        print(y)
        print(y_pred)
        # print(f'{y_pred=}')
        # print(f'{y=}')
        loss = F.mse_loss(y_pred, y).sqrt()
        self.log(f"{stage}/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
