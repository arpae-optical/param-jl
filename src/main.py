from __future__ import annotations

import argparse
import os
import random
import sys
from cProfile import label
from pathlib import Path
from typing import List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, reduce, repeat
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from sklearn.metrics import mean_squared_error
from torch import Tensor

import utils
import wandb
from backwards import BackwardModel
from data import BackwardDataModule, ForwardDataModule, StepTestDataModule
from forwards import ForwardModel
from nngraph import graph
from utils import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--forward-num-epochs",
    "--fe",
    type=int,
    default=None,
    help="Number of epochs for forward model",
)
parser.add_argument(
    "--backward-num-epochs",
    "--be",
    type=int,
    default=None,
    help="Number of epochs for backward model",
)
parser.add_argument(
    "--forward-batch-size",
    "--fbs",
    type=int,
    default=None,
    help="Batch size for forward model",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=1,
    help="How many runs for optimization",
)
parser.add_argument(
    "--backward-batch-size",
    "--bbs",
    type=int,
    default=None,
    help="Batch size for backward model",
)
parser.add_argument(
    "--prediction-iters",
    type=int,
    default=1,
    help="Number of iterations to run predictions",
)
parser.add_argument(
    "--use-cache",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load saved dataset (avoids 1 minute startup cost of fetching data from database, useful for quick tests).",
)
parser.add_argument(
    "--use-forward",
    type=eval,
    choices=[True, False],
    default=True,
    help="Whether to use a forward model at all",
)
parser.add_argument(
    "--load-forward-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained forward model. Useful for validation. Requires model to already be trained and saved.",
)
parser.add_argument(
    "--load-backward-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained backward model. Useful for validation. Requires model to already be trained and saved.",
)
args = parser.parse_args()

# The `or` idiom allows overriding values from the command line.
config: Config = {
    # "forward_lr": tune.loguniform(1e-7, 1e-4),
    "forward_lr": 1e-6,
    "backward_lr": tune.loguniform(1e-6, 1e-5),
    "latent_space_size": 224,
    "forward_num_epochs": args.forward_num_epochs or tune.choice([1600]),
    "backward_num_epochs": args.backward_num_epochs or tune.choice([2500]),
    "forward_batch_size": args.forward_batch_size or tune.choice([2**9]),
    "backward_batch_size": args.backward_batch_size or tune.choice([2**9]),
    "use_cache": True,
    "kl_coeff": tune.loguniform(2**-1, 2**0),
    "kl_variance_coeff": tune.loguniform(2**-12, 2**0),
    "prediction_iters": 1,
    "use_forward": True,
    "load_forward_checkpoint": True,
    "load_backward_checkpoint": True,
    "num_wavelens": 800,
}

concrete_config: Config = Config(
    {k: (v.sample() if hasattr(v, "sample") else v) for k, v in config.items()}
)


def main(config: Config) -> None:

    # TODO: load checkpoint for both forward and back

    true_emiss = torch.cat(
        [
            torch.tensor([1.0 for _ in range(178)]),
            torch.tensor([0.0 for _ in range(800 - 178)]),
        ]
    )

    # true_emiss = utils.step_tensor()[
    # torch.tensor([1. for _ in range )
    tpv: list[dict] = torch.load(Path("/data-new/alok/laser/minok_tpv_data.pt"))

    # TODO do for all tpv curves, not just one

    cutoff_index = 500
    for i, tpv_curve in enumerate(tpv):
        pred_emiss = tpv_curve["interp_emiss"]

        fig = plot_val(pred_emiss[:cutoff_index], true_emiss[:cutoff_index], index=178)
        fig.savefig(f"/data-new/alok/laser/figs/{i}_predicted.png", dpi=300)
        plt.close(fig)


def extend_left(tensor: Tensor, num_to_extend: int) -> Tensor:
    return torch.cat([torch.full((num_to_extend,), tensor[0]), tensor])


def plot_val(pred_emiss, true_emiss, index):
    wavelen = torch.load("/data-new/alok/laser/data.pt")["interpolated_wavelength"][0][:500]

    extended_min, extended_max = 0.1, 2.5
    granularity = 500

    extension = torch.tensor(
        [
            extended_min + (i) / granularity * (extended_max - extended_min)
            for i in range(granularity)
        ]
    )

    wavelen = torch.cat((extension, wavelen))
    pred_emiss = extend_left(pred_emiss, granularity)
    true_emiss = extend_left(true_emiss, granularity)

    fig, ax = plt.subplots()
    temp = 1400
    planck = [float(utils.planck_norm(wavelength, temp)) for wavelength in wavelen]

    planck_max = max(planck)
    planck = [wave / planck_max for wave in planck]

    wavelen_cutoff = float(wavelen[index + granularity])
    # format the predicted params
    FoMM = utils.planck_emiss_prod(wavelen, pred_emiss, wavelen_cutoff, 1400)

    step = true_emiss
    ax.plot(
        wavelen,
        pred_emiss,
        c="blue",
        alpha=0.2,
        linewidth=1.0,
        label=f"Predicted Emissivity, FoMM = {FoMM}",
    )
    ax.plot(wavelen, step, c="black", label=f"Ideal target emissivity", linewidth=2.0)
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)
