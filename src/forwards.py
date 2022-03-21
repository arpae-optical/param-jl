#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import nngraph
import wandb

import numpy as np
from pathlib import Path

from utils import Config, rmse

class ForwardModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.wavelens = torch.load(Path("wavelength.pt"))[0]
        self.config["num_wavelens"] = len(self.wavelens)
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.LazyConv1d(2**11, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2**12, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.LazyConv1d(2**13, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyBatchNorm1d(),
            nn.Flatten(),
            nn.LazyLinear(self.config["num_wavelens"]),
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
        loss = rmse(y_pred, y)
        # nngraph.emiss_error_graph(y_pred, y, "train_step.png")
        # self.log_image(key="train_forwards_error_graphs", images=["train_step.png"])

        self.log(f"forward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        randcheck = np.random.uniform()
        self.log(f"forward/val/loss", loss, prog_bar=True)
        if randcheck < 0.1:
            graph_xys = nngraph.emiss_error_graph(y_pred, y)
            graph_xs = graph_xys[6]
            graph_ys = graph_xys[4:6]
            average_RMSE = graph_xys[7]
            average_run_RMSE = graph_xys[8]
            print("randomly selected, logging image")
            print(f"loss var: {round(float(loss),5)}")
            print(f"calculated average RMSE: {round(float(average_RMSE),5)}")
            wandb.log({f"forwards_val_graph_{batch_nb}" : wandb.plot.line_series(
                    xs=graph_xs,
                    ys=graph_ys,
                    keys=[f"typical pred, RMSE ({round(float(average_run_RMSE),5)})", "typical real emiss"],
                    title=f"Typical emiss, forwards val, average RMSE {round(float(average_RMSE),5)}, loss {round(float(loss),5)}",
                    xname="wavelength")})
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x) 
        self.log(f"forward/test/loss", loss, prog_bar=True)
        randcheck = np.random.uniform()
        if randcheck < 1:
            graph_xys = nngraph.emiss_error_graph(y_pred, y)
            graph_xs = graph_xys[6]
            graph_ys = graph_xys[2:4]
            average_RMSE = graph_xys[7]
            average_run_RMSE = graph_xys[8]
            print("randomly selected, logging image")
            print(f"loss var: {round(float(loss),5)}")
            print(f"calculated average RMSE: {round(float(average_RMSE),5)}")
            wandb.log({f"forwards_test_graph_{batch_nb}" : wandb.plot.line_series(
                    xs=graph_xs,
                    ys=graph_ys,
                    keys=[f"typical pred, RMSE ({round(float(average_run_RMSE),5)})", "typical real emiss"],
                    title=f"Typical emiss, forwards test, average RMSE {round(float(average_RMSE),5)}, loss {round(float(loss),5)}",
                    xname="wavelength")})
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["forward_lr"])
