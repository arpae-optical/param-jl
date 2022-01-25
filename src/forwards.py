#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class ForwardModel(pl.LightningModule):
    def __init__(self, num_wavelens=821):
        super().__init__()
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.LazyConv1d(2 ** 11, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2 ** 12, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2 ** 13, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.Flatten(),
            nn.LazyLinear(821),
            nn.Sigmoid(),
        )

        # self.model = nn.Sequential()
        # TODO how to reverse the *data* in the Linear layers easily? transpose?
        # XXX This call *must* happen to initialize the lazy layers
        self.forward(torch.rand(2, 14, 1))

    def forward(self, x):
        # add dummy dim
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = F.huber_loss(y_pred, y)
        self.log(f"forward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = F.huber_loss(y_pred, y)
        self.log(f"forward/val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = F.huber_loss(y_pred, y)
        self.log(f"forward/test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-6)
