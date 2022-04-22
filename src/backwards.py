from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import asnumpy, parse_shape, rearrange, reduce
from einops.layers.torch import EinMix as Mix
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn
from torch.utils.data import DataLoader, TensorDataset

import data
import nngraph
import wandb
from forwards import ForwardModel
from mixer import MLPMixer
from utils import Config, Stage, rmse, split


class BackwardModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        forward_model: Optional[ForwardModel] = None,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        # self.config["num_wavelens"] = len(
        #     torch.load(Path("/data-new/alok/laser/data.pt"))["interpolated_wavelength"][0]
        # )
        if forward_model is None:
            self.forward_model = None
        else:
            self.forward_model = forward_model
            self.forward_model.freeze()

        self.trunk = nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=self.config["num_wavelens"],
                image_size=1,
                patch_size=1,
                num_classes=1_000,
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Flatten(),
        )

        self.continuous_head = nn.LazyLinear(2)
        self.discrete_head = nn.LazyLinear(12)
        # XXX this call *must* happen to initialize the lazy layers
        _dummy_input = torch.rand(2, self.config["num_wavelens"])
        self.forward(_dummy_input)

    def forward(self, x):
        h = self.trunk(x)
        laser_params = torch.sigmoid(self.continuous_head(h))
        wattages = F.one_hot(
            torch.argmax(self.discrete_head(h), dim=-1), num_classes=12
        )

        return torch.cat((laser_params, wattages), dim=-1)

    def predict_step(self, batch, _batch_nb):
        out = {"params": None, "pred_emiss": None, "pred_loss": None}
        # If step data, there's no corresponding laser params
        try:
            (y,) = batch  # y is emiss
        except ValueError:
            (y, x, uids) = batch  # y is emiss,x is laser_params
            out["true_params"] = x
            out["uids"] = uids
        out["true_emiss"] = y
        x_pred = self(y)
        out["params"] = x_pred
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            out["pred_emiss"] = y_pred
            y_loss = rmse(y_pred, y)
            out["pred_loss"] = y_loss
            loss = y_loss
        return out

    def training_step(self, batch, _batch_nb):
        # TODO: plot
        y, x, uids = (emiss, laser_params, uids) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = rmse(x_pred, x)
            self.log("backward/train/x/loss", x_loss, prog_bar=True)

        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)

            self.log(
                "backward/train/y/loss",
                y_loss,
                prog_bar=True,
            )

            loss = y_loss

        if self.current_epoch == self.config["backward_num_epochs"] - 5:
            nngraph.save_integral_emiss_point(
                y_pred,
                y,
                "/data-new/alok/laser/backwards_train_points.txt",
                all_points=True,
            )
        self.log(f"backward/train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        # TODO: plot
        y, x, uids = (emiss, laser_params, uids) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = rmse(x_pred, x)
            self.log("backward/val/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)

            self.log(
                "backward/val/y/loss",
                y_loss,
                prog_bar=True,
            )

            loss = y_loss

        if self.current_epoch == self.config["backward_num_epochs"] - 5:
            nngraph.save_integral_emiss_point(
                y_pred,
                y,
                "/data-new/alok/laser/backwards_val_points.txt",
                all_points=True,
            )
        self.log(f"backward/val/loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_nb):
        # TODO: plot
        y, x, uids = (emiss, laser_params, uids) = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = rmse(x_pred, x)
            self.log("backward/test/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)

            self.log(
                "backward/test/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss

            torch.save(x, "/data-new/alok/laser/params_true_back.pt")
            torch.save(y, "/data-new/alok/laser/emiss_true_back.pt")
            torch.save(y_pred, "/data-new/alok/laser/emiss_pred.pt")
            torch.save(x_pred, "/data-new/alok/laser/param_pred.pt")

        nngraph.save_integral_emiss_point(
            y_pred, y, "/data-new/alok/laser/backwards_test_points.txt", all_points=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["backward_lr"])
