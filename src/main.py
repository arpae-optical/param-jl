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
from scipy.interpolate import interp1d
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


GRANUALITY, CUTOFF_INDEX = 100, 4762
# GRANUALITY, CUTOFF_INDEX = 100, 4631
# GRANUALITY, CUTOFF_INDEX = 100, 4642


def nearest_idx(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`"""
    return (a - a0).abs().argmin()


def interpolate_smoothly(wavelen, emiss):
    interp_wavelen = np.linspace(min(wavelen.numpy()), max(wavelen.numpy()), num=10_000)
    interp_emiss = interp1d(wavelen().numpy(), emiss.numpy())(interp_wavelen)
    return interp_wavelen, interp_emiss


def main(config: Config) -> None:

    # TODO: load checkpoint for both forward and back

    # true_emiss = utils.step_tensor()[
    # torch.tensor([1. for _ in range )
    tpv: list[dict] = torch.load(Path("/data-new/alok/laser/minok_tpv_data.pt"))

    CUTOFF_INDEX = nearest_idx(tpv[0]["interp_wavelen"], 6.1)
    for i, tpv_curve in enumerate(tpv):
        pred_emiss = tpv_curve["interp_emiss"]
        true_emiss = torch.cat(
            [
                torch.full([CUTOFF_INDEX], 1.0),
                torch.full([len(pred_emiss) - CUTOFF_INDEX], 0.0),
            ]
        )
        # breakpoint()
        fig = plot_val(pred_emiss, true_emiss, index=CUTOFF_INDEX)
        fig.savefig(f"/data-new/alok/laser/figs/{i}_predicted.png")
        plt.close(fig)


def extend_left(tensor: Tensor, num_to_extend: int, interpolate: bool) -> Tensor:
    """interpolate: whether to do linear interpolation"""
    first_elem = tensor[0]
    if interpolate:
        left = torch.linspace(0.95, first_elem, num_to_extend)
    else:
        left = torch.full((num_to_extend,), first_elem)
    return torch.cat([left, tensor])


def plot_val(pred_emiss, true_emiss, index):
    # wavelen = torch.load("/data-new/alok/laser/data.pt")["interpolated_wavelength"][0]
    wavelen = torch.load("/data-new/alok/laser/minok_tpv_data.pt")[0]["interp_wavelen"]

    extended_min, extended_max = 0.01, 0.8333

    extension = torch.tensor(
        [
            extended_min + (i) / GRANUALITY * (extended_max - extended_min)
            for i in range(GRANUALITY)
        ]
    )

    wavelen = torch.cat((extension, wavelen))
    pred_emiss = extend_left(pred_emiss, GRANUALITY, interpolate=False)
    true_emiss = extend_left(true_emiss, GRANUALITY, interpolate=False)

    fig, ax = plt.subplots()
    temp = 1400
    planck = [float(utils.planck_norm(wavelength, temp)) for wavelength in wavelen]

    planck_max = max(planck)
    planck = [wave / planck_max for wave in planck]

    wavelen_cutoff = float(wavelen[index + GRANUALITY])
    # format the predicted params
    FoMM = utils.planck_emiss_prod(wavelen, pred_emiss, wavelen_cutoff, temp)

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
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Emissivity")
    ax.set_title(f"Measured emissivity vs. Ideal")
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)
