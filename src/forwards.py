from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import asnumpy, parse_shape, rearrange, reduce
from einops.layers.torch import EinMix as Mix
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

import nngraph
import wandb
from mixer import MLPMixer
from utils import Config, rmse


class View(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = (shape,)  # extra comma to allow handling integers as args

    def forward(self, x):
        return x.view(*self.shape)


class ForwardModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # self.config["num_wavelens"] = len(
        #     torch.load(Path("/data-new/alok/laser/data.pt"))["interpolated_wavelength"][0]
        # )
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=14,
                image_size=1,
                patch_size=1,
                num_classes=self.config["num_wavelens"],
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Sigmoid(),
        )

        # TODO how to reverse the *data* in the Linear layers easily? transpose?
        # XXX This call *must* happen to initialize the lazy layers
        self.forward(torch.rand(2, 14))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        # nngraph.emiss_error_graph(y_pred, y, "train_step.png")
        # self.log_image(key="train_forwards_error_graphs", images=["train_step.png"])
        if self.current_epoch == self.config["forward_num_epochs"] - 5:
            nngraph.save_integral_emiss_point(
                y_pred,
                y,
                "/data-new/alok/laser/forwards_train_points.txt",
                all_points=True,
            )

        self.log(f"forward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        randcheck = np.random.uniform()
        self.log(f"forward/val/loss", loss, prog_bar=True)

        if self.current_epoch > self.config["forward_num_epochs"] - 5:
            nngraph.save_integral_emiss_point(
                y_pred,
                y,
                "/data-new/alok/laser/forwards_val_points.txt",
                all_points=True,
            )
        return loss

    def test_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log(f"forward/test/loss", loss, prog_bar=True)
        nngraph.save_integral_emiss_point(
            y_pred, y, "/data-new/alok/laser/forwards_val_points.txt", all_points=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["forward_lr"])
