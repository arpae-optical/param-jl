#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data
from forwards import ForwardModel
from utils import Stage, rmse, split


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
    def __init__(self, forward_model: Optional[ForwardModel] = None):
        super().__init__()
        # self.save_hyperparameters()
        if forward_model is None:
            self.forward_model = None
        else:
            self.forward_model = forward_model
            self.forward_model.freeze()
        self.backward_model = nn.Sequential(
            nn.LazyConv1d(2 ** 11, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyConv1d(2 ** 12, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyConv1d(2 ** 13, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            # for the normalized laser params
        )
        self.continuous_head = nn.LazyLinear(3)
        self.discrete_head = nn.LazyLinear(15 - 3)
        # XXX this call *must* happen to initialize the lazy layers
        _x = self.backward_model(torch.empty(3, 935 - 1, 1))
        self.continuous_head(_x)
        self.discrete_head(_x)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        b = self.backward_model(x)
        laser_params = torch.sigmoid(self.continuous_head(b))
        wattages = F.one_hot(
            torch.argmax(self.discrete_head(b), dim=-1), num_classes=15 - 3
        )

        return torch.cat((laser_params, wattages), dim=-1)

    def training_step(self, batch, batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = F.huber_loss(x_pred, x)
        loss = x_loss
        self.log("backward/train/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = F.huber_loss(y_pred, y)

            self.log(
                "backward/train/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss
        self.log(f"backward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = F.huber_loss(x_pred, x)
        loss = x_loss
        self.log("backward/val/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = F.huber_loss(y_pred, y)
            self.log(
                "backward/val/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss
        self.log(f"backward/val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = F.huber_loss(x_pred, x)
        loss = x_loss
        self.log("backward/test/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = F.huber_loss(y_pred, y)
            self.log(
                "backward/test/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss

            torch.save(x, "params_true_back.pt")
            torch.save(y, "emiss_true_back.pt")
            torch.save(y_pred, "emiss_pred.pt")
            torch.save(x_pred, "param_pred.pt")
        self.log(f"backward/test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)
