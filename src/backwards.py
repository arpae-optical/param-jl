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

import data
from forwards import ForwardModel
from utils import Stage, split


class BackwardDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, use_cache: bool = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.use_cache = use_cache

    def setup(self, stage: Optional[str]) -> None:

        output, input = data.get_data(self.use_cache)

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


class BackwardModel(pl.LightningModule):
    def __init__(self, forward_model: ForwardModel):
        super().__init__()
        # self.save_hyperparameters()
        self.forward_model = forward_model
        self.forward_model.freeze()
        self.backward_model = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(935 - 1),
            nn.Linear(935 - 1, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 3),
            # for the normalized laser params
            nn.Sigmoid(),
        )

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
