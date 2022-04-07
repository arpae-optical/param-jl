#!/usr/bin/env python3
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
from forwards import ForwardModel
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

import data
import nngraph
import wandb
from utils import Config, Stage, rmse, split


class ForwardBackwardModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

    def forward(self, x):

        # TODO kl divergence loss for vq vae
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
        y_pred = None
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

            # if np.random.rand() < 0.1:
            #     for rand_idx in np.random.randint(
            #         0, self.config["backward_batch_size"], 3
            #     ):

            #         fig, ax = plt.subplots()
            #         ax.scatter(
            #             self.wavelens.detach().cpu(),
            #             y_pred[rand_idx].detach().cpu(),
            #             c="r",
            #         )
            #         ax.scatter(
            #             self.wavelens.detach().cpu(), y[rand_idx].detach().cpu(), c="g"
            #         )
            #         ax.set_xlabel("Wavelen (μm)")
            #         ax.set_ylabel("Emissivity")
            #         ax.legend(["Pred", "Real"])
            #         ax.set_title("Random emiss comparison")
            #         # self.logger[0].experiment.log(
            #         #     # {"global_step": self.trainer.global_step, 'chart': wandb.Image(fig)},
            #         #     {"chart": wandb.Image(fig)},
            #         # )
            #         plt.close(fig)

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
            randcheck = np.random.uniform()

            if randcheck < 0.02:
                graph_xys = nngraph.emiss_error_graph(y_pred, y)
                graph_xs = graph_xys[6]
                graph_ys = graph_xys[4:6]
                average_RMSE = graph_xys[7]
                average_run_RMSE = graph_xys[8]
                print("randomly selected, logging image")
                print(f"loss var: {round(float(loss),5)}")
                print(f"calculated average RMSE: {round(float(average_RMSE),5)}")
                wandb.log(
                    {
                        f"backwards_train_graph_batch_{_batch_nb}": wandb.plot.line_series(
                            xs=graph_xs,
                            ys=graph_ys,
                            keys=[
                                f"Average RMSE preds ({round(float(average_run_RMSE),5)})",
                                "Average RMSE real",
                            ],
                            title=f"Typical emiss, backwards val, average RMSE {round(float(rmse(y,y_pred)),5)}, loss {round(float(loss),5)}",
                            xname="wavelength",
                        )
                    }
                )

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
        kl_loss = 0
        y_pred = None
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
        randcheck = np.random.uniform()
        if randcheck < 0.02:
            print("randomly selected back, logging image")
            graph_xys = nngraph.emiss_error_graph(y_pred, y)
            graph_xs = graph_xys[6]
            graph_ys = graph_xys[2:4]
            average_RMSE = graph_xys[7]
            average_run_RMSE = graph_xys[8]
            print(f"loss: {round(float(loss),5)}")
            print(f"calculated average RMSE: {round(float(average_RMSE),5)}")
            wandb.log(
                {
                    f"backwards_val_graph_{batch_nb}": wandb.plot.line_series(
                        xs=graph_xs,
                        ys=graph_ys,
                        keys=[
                            f"typical pred, RMSE ({round(float(average_run_RMSE),5)})",
                            "typical real emiss",
                        ],
                        title=f"Typical emiss, backwards val, average RMSE {round(float(rmse(y,y_pred)),5)}, loss {round(float(loss),5)}",
                        xname="wavelength",
                    )
                }
            )
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
        kl_loss = 0
        y_pred = None
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
        randcheck = np.random.uniform()
        if randcheck < 1:
            print("randomly selected back, logging image")
            graph_xys = nngraph.emiss_error_graph(y_pred, y)
            graph_xs = graph_xys[6]
            graph_ys = graph_xys[2:4]
            average_RMSE = graph_xys[7]
            average_run_RMSE = graph_xys[8]
            print(f"loss: {round(float(loss),5)}")
            print(f"calculated average RMSE: {round(float(average_RMSE),5)}")
            wandb.log(
                {
                    f"backwards_test_graph_{batch_nb}": wandb.plot.line_series(
                        xs=graph_xs,
                        ys=graph_ys,
                        keys=[
                            f"typical pred, RMSE ({round(float(average_run_RMSE),5)})",
                            "typical real emiss",
                        ],
                        title=f"Typical emiss, backwards test, average RMSE {round(float(rmse(y,y_pred)),5)}, loss {round(float(loss),5)}",
                        xname="wavelength",
                    )
                }
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["backward_lr"])