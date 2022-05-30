from __future__ import annotations

import argparse
import os
import random
import sys
from cProfile import label
from pathlib import Path
from typing import List, Optional, TypedDict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
from nngraph import training_set_mean_vs_stdev
from scipy.interpolate import interp1d
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

    forward_trainer = pl.Trainer(
        max_epochs=config["forward_num_epochs"],
        logger=[
            WandbLogger(
                name="Forward laser params",
                save_dir="/data-new/alok/laser/wandb_logs/forward",
                offline=False,
                project="Laser Forward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data-new/alok/laser/test_tube_logs/forward",
                name="Forward",
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="forward/val/loss",
                dirpath="/data-new/alok/laser/weights/forward",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=2),
        ],
        gpus=1,
        precision=32,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        check_val_every_n_epoch=min(3, config["forward_num_epochs"] - 1),
        gradient_clip_val=0.5,
        log_every_n_steps=min(3, config["forward_num_epochs"] - 1),
    )

    backward_trainer = pl.Trainer(
        max_epochs=config["backward_num_epochs"],
        logger=[
            WandbLogger(
                name="Backward laser params",
                save_dir="/data-new/alok/laser/wandb_logs/backward",
                offline=False,
                project="Laser Backward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data-new/alok/laser/test_tube_logs/backward", name="Backward"
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="backward/val/loss",
                dirpath="/data-new/alok/laser/weights/backward",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=10),
        ],
        gpus=1,
        precision=32,
        weights_summary="full",
        check_val_every_n_epoch=min(3, config["backward_num_epochs"] - 1),
        gradient_clip_val=0.5,
        log_every_n_steps=min(3, config["backward_num_epochs"] - 1),
    )

    backward_datamodule = BackwardDataModule(config)

    # TODO: load checkpoint for both forward and back
    forward_model = ForwardModel(config)
    backward_model = BackwardModel(config=config, forward_model=forward_model)


    out = backward_trainer.predict(
        model=backward_model,
        ckpt_path="/data-new/alok/laser/weights/backward/laser-april-18.ckpt",
        datamodule=backward_datamodule,
        return_predictions=True,
    )[0]
    true_emiss = out["true_emiss"]
    pred_array = []
    breakpoint()
    param_csv = open("/home/collin/param-jl/src/resampled_big.csv", "r")
    uid_list = [line[0:5] for line in param_csv]
    watt_list = [line[7] for line in param_csv]
    speed_list = [line[9:18] for line in param_csv]
    spacing_list = [line[19:27] for line in param_csv]
    param_csv.close()
    RMSE_list = []
    original_data = torch.load("/data-new/alok/laser/data.pt")
    y, stdevs = training_set_mean_vs_stdev()
    for uid in range(29100, 32250):
        try:
            real_index = int((out["uids"] == int(uid)+0).nonzero()[0])
            manufactured_txt = open(f"src/minok_real_by_UID/UID_{uid} .txt")
            man_x = [float(line[3:16]) for line in manufactured_txt]
            manufactured_txt.close()
            manufactured_txt = open(f"src/minok_real_by_UID/UID_{uid} .txt")
            man_y = [float(line[17:]) for line in manufactured_txt]
            uids = original_data["uids"]
            norm_laser = original_data["normalized_laser_params"]
            f = interp1d(man_x, man_y)
            
            pred_emiss = out["pred_emiss"][real_index]
            fig, RMSE = plot_val(y, true_emiss[real_index], f)
            fig.savefig(f"/data-new/alok/laser/figs/UID_verify_{uid}_predicted.png", dpi=300)
            RMSE_list.append(RMSE)
            plt.close(fig)
            manufactured_txt.close()
        except:
            None
    print(RMSE_list)
    print(np.mean(RMSE_list))
    print(np.std(RMSE_list))
            


def plot_val(pred_emiss, true_emiss, manufactured_f):
    wavelen = torch.load("/data-new/alok/laser/data.pt")["interpolated_wavelength"][0]
    
    
    fig, ax = plt.subplots()
    ax.plot(
        wavelen,
        pred_emiss,
        c="blue",
        alpha=0.5,
        linewidth=1.0,
        label=f"Predicted Emissivity",
    )

    ax.plot(
        wavelen,
        [manufactured_f(emiss) for emiss in wavelen],
        c="green",
        alpha=0.5,
        linewidth=1.0,
        label=f"Manufactured Emissivity",
    )

    ax.plot(
        wavelen,
        true_emiss,
        c="red",
        alpha=0.5,
        linewidth=1.0,
        label=f"Original Validation Emissivity",
    )
    
    

    ax.legend()
    return fig, mean_squared_error(pred_emiss,[manufactured_f(emiss) for emiss in wavelen], squared = False)


if __name__ == "__main__":
    main(concrete_config)