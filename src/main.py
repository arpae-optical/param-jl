from __future__ import annotations

import argparse
import os
import random
import sys
from cProfile import label
from pathlib import Path
from typing import List, Optional, TypedDict
from xml.sax.saxutils import prepare_input_source

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
    param_csv = open("/home/collin/param-jl/src/resampled_big.csv", "r")
    uid_list = [line[0:5] for line in param_csv]
    watt_list = [line[7] for line in param_csv]
    speed_list = [line[9:18] for line in param_csv]
    spacing_list = [line[19:27] for line in param_csv]
    param_csv.close()
    RMSE_list = []
    original_data = torch.load("/data-new/alok/laser/data.pt")
    y, stdevs = training_set_mean_vs_stdev()
    laser_array = []
    uid_list = []
    uid_i_list = []
    emiss_array = []
    man_emiss_array = []
    manufactured_txt = open(f"src/minok_real_by_UID/UID_{32248} .txt")
    man_x = [float(line[3:16]) for line in manufactured_txt]
    manufactured_txt.close()

    for uid in range(29100, 32250):
        try:
            manufactured_txt = open(f"src/minok_real_by_UID/UID_{uid} .txt")
            man_y = [float(line[17:]) for line in manufactured_txt]
            uids = original_data["uids"]
            uid_index = int((uids == uid).nonzero(as_tuple = True)[0])
            norm_laser = original_data["normalized_laser_params"][uid_index]
            interp_emiss = original_data["interpolated_emissivity"][uid_index]
            f = interp1d(man_x, man_y)
            man_interp_emiss = [f((i+1)*12/800+2.5) for i in range(800)]
            man_emiss_array.append(man_interp_emiss)
            laser_array.append(norm_laser)
            
            emiss_array.append(interp_emiss)
            uid_list.append(uid)
            uid_i_list.append(uid_index)
            manufactured_txt.close()
        except:
            None
    
    laser_array = torch.stack(laser_array)
    emiss_array = torch.stack(emiss_array)
    print(laser_array.size())
    target = torch.zeros(512, 800)
    target[:100, :] = emiss_array
    pred_param = backward_model(target)
    pred_emiss = forward_model(pred_param)

    graph(True, True, laser_array, emiss_array, pred_param, man_emiss_array, pred_param, pred_emiss)

    end_file = open(f"src/orig_data.csv", "w")
    end_file.write(f"UID, Speed, Spacing, Wattage 0.2, Wattage 0.3, Wattage 0.4, Wattage 0.5, Wattage 0.6, Wattage 0.7, Wattage 0.8, Wattage 0.9, Wattage 1.0, Wattage 1.1, Wattage 1.2, Wattage 1.3, Predicted Emissivity (800 indices)")
    end_file.write(f"\n")
    for i in range(100):
        end_file.write(f"{uid_list[i]}, ")
        for j in range(14):
            end_file.write(f"{laser_array[i][j]}, ")
        for k in range(800):
            end_file.write(f"{emiss_array[i][k]}, ")
        end_file.write(f"\n")
    end_file.close()

    end_file = open(f"src/pred_data.csv", "w")
    end_file.write(f"UID, Speed, Spacing, Wattage 0.2, Wattage 0.3, Wattage 0.4, Wattage 0.5, Wattage 0.6, Wattage 0.7, Wattage 0.8, Wattage 0.9, Wattage 1.0, Wattage 1.1, Wattage 1.2, Wattage 1.3, Predicted Emissivity (800 indices)")
    end_file.write(f"\n")
    for i in range(100):
        end_file.write(f"{uid_list[i]}, ")
        for j in range(14):
            end_file.write(f"{pred_param[i][j]}, ")
        for k in range(800):
            end_file.write(f"{pred_emiss[i][k]}, ")
        end_file.write(f"\n")
    end_file.close()

    end_file = open(f"src/man_data.csv", "w")
    end_file.write(f"UID, Speed, Spacing, Wattage 0.2, Wattage 0.3, Wattage 0.4, Wattage 0.5, Wattage 0.6, Wattage 0.7, Wattage 0.8, Wattage 0.9, Wattage 1.0, Wattage 1.1, Wattage 1.2, Wattage 1.3, Predicted Emissivity (800 indices)")
    end_file.write(f"\n")
    for i in range(100):
        end_file.write(f"{uid_list[i]}, ")
        for j in range(14):
            end_file.write(f"{pred_param[i][j]}, ")
        for k in range(800):
            end_file.write(f"{man_emiss_array[i][k]}, ")
        end_file.write(f"\n")
    end_file.close()



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