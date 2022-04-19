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
    #original_pred = out["pred_emiss"]
    original_pred = forward_model(backward_model(true_emiss.detach())).detach()
    pred_array = []
    variant_num = 3
    for i in range(variant_num):
        new_true = [torch.tensor(emiss+random.uniform(-0.03, 0.03)) for emiss in true_emiss]
        new_true = torch.stack(new_true)
        back = backward_model(new_true)
        new_pred = forward_model(back)
        pred_array.append(new_pred.detach())
        print("mse between original pred and noise-added truth")
        print(mean_squared_error(original_pred, new_true.detach()))
        print("mse between new prediction and original truth")
        print(mean_squared_error(true_emiss.detach(), new_pred.detach()))

    for i in range(0, 100, 5):

        pred_emiss = []
        for j in range(variant_num):
            pred_emiss.append(pred_array[j][i])
        pred_emiss = torch.stack(pred_emiss)
        fig = plot_val(pred_emiss, true_emiss[i], original_pred[i], uids[i])
        fig.savefig(f"/data/alok/laser/figs/{i}_predicted.png", dpi=300)
        breakpoint()
        plt.close(fig)


        fig2 = plot_val(train_emiss, true_emiss[i], original_pred[i], uids[i])
        fig2.savefig(f"/data/alok/laser/figs/{i}_train_set.png", dpi=300)
        plt.close(fig2)
    



def plot_val(pred_emiss, true_emiss, original_pred, uid):
    wavelen = torch.load("/data/alok/laser/data.pt")["interpolated_wavelength"][0]
    print("mse between true and pred")
    print(pred_emiss)
    print(true_emiss)
    print(mean_squared_error(true_emiss.detach(), pred_emiss.detach()[1]))
    mean, std = (
        pred_emiss.mean(0),
        pred_emiss.std(0),
    )

    
    fig, ax = plt.subplots() 
    ax.plot(wavelen, true_emiss, label = "true")
    ax.plot(wavelen, mean, label = "pred/train mean")
    ax.plot(wavelen, original_pred, label = "original pred")
    ax.set_xlabel("wavelen")
    ax.set_ylabel("emiss")
    ax.set_title(str(uid))
    ax.set_ylim(0, 1)
    plt.fill_between(wavelen, mean - std, mean + std, alpha=0.5)
    ax.legend()
    return fig


if __name__ == "__main__":
    main(concrete_config)