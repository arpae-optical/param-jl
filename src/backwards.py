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

from forwards import ForwardDataModule, ForwardModel, get_data
from utils import Stage, split

parser = argparse.ArgumentParser()

parser.add_argument("--num-epochs", "-n", type=int, default=100_000)
parser.add_argument("--batch-size", "-b", type=int, default=2 ** 10)
args = parser.parse_args()


class BackwardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = args.batch_size,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str]) -> None:

        output, input = get_data()

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


class BackwardModel(pl.LightningModule):
    def __init__(self, forward_model):
        super().__init__()
        # self.save_hyperparameters()
        self.forward_model = forward_model
        self.forward_model.freeze()
        self.backward_model = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(935),
            nn.Linear(935, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 3),
        )
        # TODO how to reverse the *data* in the Linear layers easily? transpose?

        # TODO add mode arg

    def forward(self, x):
        return self.backward_model(x)

    def training_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="train")

    def validation_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="test")

    def _step(self, batch, batch_nb, stage: Stage):
        y, x = batch
        x_pred = self(y)
        y_pred = self.forward_model(x_pred)
        y_loss = F.mse_loss(y_pred, y).sqrt()
        x_loss = F.mse_loss(x_pred, x).sqrt()
        loss = x_loss + y_loss
        self.log(f"{stage}/x/loss", x_loss)
        self.log(f"{stage}/y/loss", y_loss)
        self.log(f"{stage}/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


forward_trainer = pl.Trainer(
    max_epochs=10_000,
    logger=[
        WandbLogger(
            name="Forward laser params",
            save_dir="wandb_logs/forward",
            offline=False,
            project="Laser",
            log_model=True,
        ),
        TestTubeLogger(
            save_dir="test_tube_logs/forward", name="Forward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="weights/forward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    overfit_batches=1,
    track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
)

forward_model = ForwardModel()
forward_data_module = ForwardDataModule(batch_size=3)
forward_trainer.fit(forward_model, datamodule=forward_data_module)


backward_trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    logger=[
        WandbLogger(
            name="Backward laser params",
            save_dir="wandb_logs/backward",
            offline=False,
            project="Laser",
            log_model=True,
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
    overfit_batches=1,
    track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
)


backward_model = BackwardModel(forward_model=forward_model)
backward_data_module = BackwardDataModule()
backward_trainer.fit(backward_model, datamodule=backward_data_module)
