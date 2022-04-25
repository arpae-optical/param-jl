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
import torch.nn.functional as F
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
from utils import Config, rmse

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
    "forward_num_epochs": args.forward_num_epochs or tune.choice([1600]),
    "backward_num_epochs": args.backward_num_epochs or tune.choice([2500]),
    "forward_batch_size": args.forward_batch_size or tune.choice([2**9]),
    "backward_batch_size": args.backward_batch_size or tune.choice([2**9]),
    "use_cache": True,
    "use_forward": True,
    "load_forward_checkpoint": True,
    "load_backward_checkpoint": True,
    "num_wavelens": 800,
}

concrete_config: Config = Config(
    {k: (v.sample() if hasattr(v, "sample") else v) for k, v in config.items()}
)

WATTAGE_IDXS = {
    0.2: 0,
    0.3: 1,
    0.4: 2,
    0.5: 3,
    0.6: 4,
    0.7: 5,
    0.8: 6,
    0.9: 7,
    1.0: 8,
    1.1: 9,
    1.2: 10,
    1.3: 11,
}


def main(config: Config) -> None:

    # TODO: load checkpoint for both forward and back

    # true_emiss = utils.step_tensor()[
    # torch.tensor([1. for _ in range )
    all_true_emiss: list[dict] = torch.load(
        Path("/data-new/alok/laser/minok_laser_preds.pt")
    )
    MAX_SPEED = 700.0
    MAX_SPACING = 42.0
    forward_model = ForwardModel.load_from_checkpoint(
        "/data-new/alok/laser/weights/forward/laser-april-18.ckpt"
    )
    with torch.no_grad():
        for i, entry in enumerate(all_true_emiss):
            # ['raw_emiss', 'interp_emiss', 'interp_wavelen', 'raw_wavelen', 'power', 'speed', 'spacing'])
            # unsqueeze to avoid concat error
            speed, spacing, power = entry["speed"], entry["spacing"], entry["power"]
            true_emiss = entry["interp_emiss"]

            normalized_speed = torch.as_tensor(speed / MAX_SPEED).unsqueeze(0)
            normalized_spacing = torch.as_tensor(spacing / MAX_SPACING).unsqueeze(0)
            one_hot_power = F.one_hot(
                torch.tensor(WATTAGE_IDXS[round(power, 1)]),
                num_classes=len(WATTAGE_IDXS),
            )
            # unsqueeze to add batch dimension
            laser_params = torch.cat(
                [normalized_speed, normalized_spacing, one_hot_power]
            ).unsqueeze(0)
            pred_emiss = forward_model(laser_params).squeeze(0)
            err = rmse(pred_emiss, true_emiss)
            print(f"{err = }")
            fig = plot_fig(
                speed,
                spacing,
                power,
                entry["interp_wavelen"],
                pred_emiss,
                true_emiss,
                err,
            )
            fig.savefig(Path(f"/data-new/alok/laser/figs/minok-val-{i}.png"))
            plt.close(fig)


def plot_fig(speed, spacing, power, wavelen, pred_emiss, true_emiss, err):
    fig, ax = plt.subplots()
    ax.plot(wavelen, true_emiss, label="true", c="g")
    ax.plot(wavelen, pred_emiss, label="pred", c="r")
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Emissivity")
    ax.set_ylim(0,1)
    ax.set_title(
        f"RMSE = {err:.3f} (Speed = {speed} mm/s, Spacing = {spacing} μm, Power = {power} W)"
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)
