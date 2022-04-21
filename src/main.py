from __future__ import annotations

import argparse
from cProfile import label
import os
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from einops import rearrange, reduce, repeat
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from torch import Tensor

import wandb
from backwards import BackwardModel
from data import BackwardDataModule, ForwardDataModule, StepTestDataModule
from forwards import ForwardModel
from nngraph import graph
from utils import Config

import random
import numpy as np
from sklearn.metrics import mean_squared_error

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
                save_dir="/data/alok/laser/wandb_logs/forward",
                offline=False,
                project="Laser Forward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data/alok/laser/test_tube_logs/forward",
                name="Forward",
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="forward/val/loss",
                dirpath="/data/alok/laser/weights/forward",
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
                save_dir="/data/alok/laser/wandb_logs/backward",
                offline=False,
                project="Laser Backward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="/data/alok/laser/test_tube_logs/backward", name="Backward"
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="backward/val/loss",
                dirpath="/data/alok/laser/weights/backward",
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

    forward_data_module = ForwardDataModule(config)
    backward_data_module = BackwardDataModule(config)

    # TODO: load checkpoint for both forward and back
    forward_model = ForwardModel(config)
    backward_model = BackwardModel(config=config, forward_model=forward_model)

    pred_iters = config["prediction_iters"]
    latent = config["latent_space_size"]
    variance = config["kl_variance_coeff"]

    out = backward_trainer.predict(
        model=backward_model,
        ckpt_path="/data/alok/laser/weights/backward/last.ckpt",
        datamodule=backward_data_module,
        return_predictions=True,
    )[0]
    uids = out["uids"]
    train_emiss = torch.load("/data/alok/laser/data.pt")["interpolated_emissivity"]

    true_emiss = out["true_emiss"]
    pred_array = []
    variant_num = 10
    arbitrary_list = range(0, 100, 5)
    uid_list = [[] for i in range(variant_num)]
    watt_list = [[] for i in range(variant_num)]
    speed_list = [[] for i in range(variant_num)]
    spacing_list = [[] for i in range(variant_num)]
    random_emissivities_list = [[] for i in range(variant_num)]
    param_std_total = 0
    for i in range(variant_num):
        #new_true = [torch.tensor(emiss+random.uniform(-0.05, 0.05)) for emiss in true_emiss]
        random_mult = random.uniform(-0.3, 0.3)
        new_true = torch.clamp(torch.tensor([[(random_mult*(1/emiss)*(e_index/3+100)/600)*emiss+emiss for e_index, emiss in enumerate(sub_emiss)] for sub_emiss in true_emiss]),0,1)
        
        if i == 0:
            new_true = true_emiss
        back = backward_model(new_true)
        #add spacing
        space = back.detach()[:, 0]

        #add speed
        speed = back.detach()[:, 1]
        #add watt
        watt2 = [(np.where(watt1 == 1)[0]+1)/10 for watt1 in np.array(back[:, 2:].detach())]
        
        for j in arbitrary_list:
            watt_list[i].append(watt2[j])
            speed_list[i].append(speed[j]*690+10)
            spacing_list[i].append(space[j]*41+1)
            random_emissivities_list[i].append(random_mult)

    
        #minspeed = 10, maxspeed = 700

        #min 1 max 42

        new_pred = forward_model(back)
        
        pred_array.append(new_pred.detach())
    
    for i, j in enumerate(arbitrary_list):
        param_std_total += np.std((np.array(spacing_list)[:, i]-1)/41)

        param_std_total += np.std((np.array(speed_list)[:, i]-10)/690)

        param_std_total += np.std(np.array(watt_list)[:, i])
        
    print(param_std_total)
    

    print("average std across params:")
    print(param_std_total/variant_num/3)
    param_file = open("resampled_random_emissivities.txt", "a")
    for random_emissivities in random_emissivities_list:
        for random_emissivitie in random_emissivities:
            param_file.write("%.5f" % round(float(random_emissivitie), 5)+", ")
        
        param_file.write("\n")
    param_file.close()

    param_file = open("resampled_watt.txt", "a")
    for watts in watt_list:
        for watt in watts:
            param_file.write("%.5f" % round(float(watt), 5)+", ")
        
        param_file.write("\n")
    param_file.close()

    param_file = open("resampled_speed.txt", "a")
    for speeds in speed_list:
        for speed in speeds:
            param_file.write("%.5f" % round(float(speed), 5)+", ")
        
        param_file.write("\n")
    param_file.close()

    param_file = open("resampled_spacing.txt", "a")
    for spacings in spacing_list:
        for spacing in spacings:
            param_file.write("%.5f" % round(float(spacing), 5)+", ")
        
        param_file.write("\n")
    param_file.close()

    for i in arbitrary_list:

        pred_emiss = []
        for j in range(variant_num):
            pred_emiss.append(pred_array[j][i])
        pred_emiss = torch.stack(pred_emiss)
        fig = plot_val(pred_emiss, true_emiss[i], i, True)
        fig.savefig(f"/data/alok/laser/figs/{i}_predicted.png", dpi=300)
        plt.close(fig)


        fig2 = plot_val(train_emiss, true_emiss[i], i, False, stdevs = 1)
        fig2.savefig(f"/data/alok/laser/figs/{i}_train_set.png", dpi=300)
        plt.close(fig2)


def plot_val(pred_emiss, true_emiss, uid, pred_set_flag, stdevs = 1):
    wavelen = torch.load("/data/alok/laser/data.pt")["interpolated_wavelength"][0]
    mean, std = (
        pred_emiss.mean(0),
        pred_emiss.std(0),
    )
    label_string = None
    if pred_set_flag == True:
        label_string = "Predicted"
    if pred_set_flag == False:
        label_string = "Train Set"

    
    fig, ax = plt.subplots() 
    ax.plot(wavelen, true_emiss, label = "True emissivity")
    ax.plot(wavelen, mean, label = f"{label_string} emissivity, {stdevs} stdev above and below", linewidth = 1)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Emissivity")
    ax.set_title(f"Mean {label_string} Emissivity, UID {uid}", fontsize = 12)
    ax.set_ylim(0, 1)
    plt.fill_between(wavelen, mean - std*stdevs, mean + std*stdevs, alpha=0.5)
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)
        
    
    
    

