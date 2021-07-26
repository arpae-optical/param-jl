#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data
from utils import Stage, split,rmse


class ForwardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.use_cache = use_cache

    def setup(self, stage: Optional[str]) -> None:

        input, output = data.get_data(self.use_cache)
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
            nn.LazyConv1d(2 ** 11, kernel_size=1),
            nn.GELU(),
            nn.Dropout(.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2 ** 12, kernel_size=1),
            nn.GELU(),
            nn.Dropout(.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2 ** 13, kernel_size=1),
            nn.GELU(),
            nn.Dropout(.5),
            nn.LazyBatchNorm1d(),
            nn.Flatten(),
            nn.LazyLinear(935 - 1),
            nn.Sigmoid(),
        )
        # TODO use convnet
        # self.model = nn.Sequential()
        # TODO how to reverse the *data* in the Linear layers easily? transpose?
        # XXX This call *must* happen to initialize the lazy layers
        self.model(torch.empty(3, 16, 1))

    def forward(self, x):
        # add dummy dim
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log(f"forward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log(f"forward/val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log(f"forward/test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)
