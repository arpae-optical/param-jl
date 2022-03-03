#!/usr/bin/env python3
from __future__ import annotations

import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

import data
from forwards import ForwardModel
from utils import Config, Stage, rmse, split


class BackwardDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["backward_batch_size"]

    def setup(self, stage: Optional[str]) -> None:

        output, input = data.get_data(self.config["use_cache"])

        splits = split(len(input))

        self.train, self.val, self.test = [
            TensorDataset(
                input[splits[s].start : splits[s].stop],
                output[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )


class BackwardModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        forward_model: Optional[ForwardModel] = None,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        if forward_model is None:
            self.forward_model = None
        else:
            self.forward_model = forward_model
            self.forward_model.freeze()

        self.encoder = nn.Sequential(
            nn.LazyConv1d(2**11, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyConv1d(2**12, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LazyConv1d(2**13, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Flatten(),
        )

        Z = self.config["latent_space_size"]
        self.mean_head = nn.LazyLinear(Z)
        self.log_var_head = nn.LazyLinear(Z)

        self.decoder = nn.Sequential(
            nn.LazyLinear(512),
            nn.GELU(),
            nn.LazyLinear(256),
        )
        self.continuous_head = nn.LazyLinear(2)
        self.discrete_head = nn.LazyLinear(12)
        # XXX this call *must* happen to initialize the lazy layers
        # TODO fix
        _dummy_input = torch.rand(2, self.config["num_wavelens"], 1)
        self.forward(_dummy_input)

    def forward(self, x, mode: Stage = "train"):
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        h = self.encoder(x)
        mean, log_var = self.mean_head(h), self.log_var_head(h)

        std = (log_var / 2).exp()

        dist = Normal(loc=mean, scale=std)
        zs = dist.rsample()

        decoded = self.decoder(zs)

        laser_params = torch.sigmoid(self.continuous_head(decoded))
        wattages = F.one_hot(
            torch.argmax(self.discrete_head(decoded), dim=-1), num_classes=12
        )

        return {
            "params": torch.cat((laser_params, wattages), dim=-1),
            "dist": dist,
        }

    def predict_step(self, batch, _batch_nb):
        out = {"params": [], "pred_emiss": [], "pred_loss": []}
        # If step data, there's no corresponding laser params
        try:
            (y,) = batch  # y is emiss
        except ValueError:
            (y, x) = batch  # y is emiss,x is laser_params
            out["true_params"] = x
        out["true_emiss"] = y
        for pred in [self(y) for _ in range(50)]:
            x_pred, dist = pred["params"], pred["dist"]
            out["params"].append(x_pred)
            if self.forward_model is not None:
                y_pred = self.forward_model(x_pred)
                out["pred_emiss"].append(y_pred)
                y_loss = rmse(y_pred, y)
                out["pred_loss"].append(y_loss)
                kl_loss = (
                    self.config["kl_coeff"]
                    * kl_divergence(
                        dist,
                        Normal(
                            torch.zeros_like(dist.mean),
                            self.config["kl_variance_coeff"]
                            * torch.ones_like(dist.variance),
                        ),
                    ).mean()
                )
                loss = y_loss + kl_loss
                print(f"Predicted loss: {loss}")
        return out

    def training_step(self, batch, _batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        x_pred, dist = x_pred["params"], x_pred["dist"]
        if self.forward_model is None:
            with torch.no_grad():
                x_loss = rmse(x_pred, x)
        else:
            x_loss = rmse(x_pred, x)
        loss = x_loss
        self.log("backward/train/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)
            kl_loss = (
                self.config["kl_coeff"]
                * kl_divergence(
                    dist,
                    Normal(
                        torch.zeros_like(dist.mean),
                        self.config["kl_variance_coeff"]
                        * torch.ones_like(dist.variance),
                    ),
                ).mean()
            )
            self.log(
                "backward/train/kl/loss",
                kl_loss,
                prog_bar=True,
            )

            self.log(
                "backward/train/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss + kl_loss
        self.log(f"backward/train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        x_pred, dist = x_pred["params"], x_pred["dist"]
        if self.forward_model is None:
            with torch.no_grad():
                x_loss = rmse(x_pred, x)
        else:
            x_loss = rmse(x_pred, x)
        loss = x_loss
        self.log("backward/val/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)

            kl_loss = (
                self.config["kl_coeff"]
                * kl_divergence(
                    dist,
                    Normal(
                        torch.zeros_like(dist.mean),
                        self.config["kl_variance_coeff"]
                        * torch.ones_like(dist.variance),
                    ),
                ).mean()
            )
            self.log(
                "backward/train/kl/loss",
                kl_loss,
                prog_bar=True,
            )

            self.log(
                "backward/val/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss + kl_loss
        self.log(f"backward/val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        # TODO: plot
        y, x = (emiss, laser_params) = batch

        x_pred = self(y)
        x_pred, dist = x_pred["params"], x_pred["dist"]
        if self.forward_model is None:
            with torch.no_grad():
                x_loss = rmse(x_pred, x)
        else:
            x_loss = rmse(x_pred, x)
        loss = x_loss
        self.log("backward/test/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)
            kl_loss = (
                self.config["kl_coeff"]
                * kl_divergence(
                    dist,
                    Normal(
                        torch.zeros_like(dist.mean),
                        self.config["kl_variance_coeff"]
                        * torch.ones_like(dist.variance),
                    ),
                ).mean()
            )
            self.log(
                "backward/train/kl/loss",
                kl_loss,
                prog_bar=True,
            )

            self.log(
                "backward/test/y/loss",
                y_loss,
                prog_bar=True,
            )
            loss = y_loss + kl_loss

            torch.save(x, "params_true_back.pt")
            torch.save(y, "emiss_true_back.pt")
            torch.save(y_pred, "emiss_pred.pt")
            torch.save(x_pred, "param_pred.pt")
        self.log(f"backward/test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["backward_lr"])
