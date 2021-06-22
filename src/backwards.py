#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence

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

from utils import split

parser = argparse.ArgumentParser()

parser.add_argument("--num-epochs", "-n", type=int, default=1_000_000)
parser.add_argument("--batch-size", "-b", type=int, default=2 ** 10)
args = parser.parse_args()

client = pymongo.MongoClient(
    "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
)
db = client.propopt.laser_samples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = args.batch_size,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str]) -> None:

        input, output = [], []

        for entry in tqdm(db.find()):
            emiss_plot: List[float] = [
                ex["normal_emissivity"] for ex in entry["emissivity_spectrum"]
            ]
            # drop all problematic emiss (only 3% of data dropped)
            if len(emiss_plot) != 935:
                continue

            output.append(
                [
                    entry["laser_scanning_speed_x_dir_mm_per_s"],
                    entry["laser_scanning_line_spacing_y_dir_micron"],
                    float(entry["laser_repetition_rate_kHz"]),
                ]
            )
            input.append(emiss_plot)

        input, output = torch.FloatTensor(input), torch.FloatTensor(output)
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


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(935, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 3),
        )
        # TODO how to reverse the *data* in the Linear layers easily? transpose?

        # TODO add mode arg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    logger=[
        WandbLogger(
            name="Backward laser params",
            save_dir="wandb_logs/backward",
            offline=False,
            project="Laser",
            log_model=True,
            sync_step=True,
        ),
        TestTubeLogger(
            save_dir="test_tube_logs/backward", name="Backward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="weights/backward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
)

model = Model()
data_module = DataModule()
trainer.fit(model, datamodule=data_module)
