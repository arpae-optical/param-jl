#!/usr/bin/env python3

from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path
from typing import List, Optional, TypedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from torch import Tensor

from backwards import BackwardModel
from data import BackwardDataModule, ForwardDataModule, StepTestDataModule
from forwards import ForwardModel
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


def main(config: Config) -> None:
    ray_plugin = RayPlugin(num_workers=1, num_cpus_per_worker=64, use_gpu=True)

    forward_trainer = pl.Trainer(
        max_epochs=config["forward_num_epochs"],
        logger=[
            WandbLogger(
                name="Forward laser params",
                save_dir="wandb_logs/forward",
                offline=False,
                project="Laser Forward",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="test_tube_logs/forward",
                name="Forward",
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="forward/val/loss",
                dirpath="weights/forward",
                save_top_k=1,
                mode="min",
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=100),
            TuneReportCallback(
                {"forward_val_loss": "forward/val/loss"}, on="validation_end"
            ),
        ],
        plugins=[ray_plugin],
        gpus=1,
        precision=32,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        check_val_every_n_epoch=10,
        gradient_clip_val=0.5,
        log_every_n_steps=min(10, config["forward_num_epochs"]),
    )

    backward_trainer = pl.Trainer(
        max_epochs=config["backward_num_epochs"],
        logger=[
            WandbLogger(
                name="Backward laser params",
                save_dir="wandb_logs/backward",
                offline=False,
                project="Laser Backward",
                log_model=True,
            ),
            TensorBoardLogger(save_dir="test_tube_logs/backward", name="Backward"),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="backward/val/loss",
                dirpath="weights/backward",
                save_top_k=1,
                mode="min",
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=10),
            TuneReportCallback(
                {"backward_val_loss": "backward/val/loss"}, on="validation_end"
            ),
        ],
        plugins=[ray_plugin],
        gpus=1,
        precision=32,
        weights_summary="full",
        check_val_every_n_epoch=10,
        gradient_clip_val=0.5,
        log_every_n_steps=min(30, config["backward_num_epochs"]),
    )

    forward_data_module = ForwardDataModule(config)
    backward_data_module = BackwardDataModule(config)
    step_data_module = StepTestDataModule()

    # TODO: load checkpoint for both forward and back
    if config["use_forward"]:
        forward_model = ForwardModel(config)
        if not config["load_forward_checkpoint"]:
            forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

        forward_trainer.test(
            model=forward_model,
            ckpt_path="best",
            datamodule=forward_data_module,
        )
        backward_model = BackwardModel(config=config, forward_model=forward_model)
    else:
        backward_model = BackwardModel(config=config, forward_model=None)

    if not config["load_backward_checkpoint"]:
        backward_trainer.fit(model=backward_model, datamodule=backward_data_module)
    backward_trainer.test(
        model=backward_model,
        ckpt_path="best",
        datamodule=backward_data_module,
    )

    for i in range(config["prediction_iters"]):
        preds: List[Tensor] = backward_trainer.predict(
            model=backward_model,
            ckpt_path="best",
            datamodule=backward_data_module,
            return_predictions=True,
        )
        torch.save(preds, f"src/preds_i_validation/preds_{i}_validation")


# The `or` idiom allows overriding values from the command line.
config: Config = {
    "forward_lr": tune.loguniform(1e-7, 1e-4),
    "backward_lr": tune.loguniform(1e-7, 1e-4),
    "latent_space_size": tune.qlograndint(2**5, 2**10, 1),
    "forward_num_epochs": args.forward_num_epochs or tune.choice([3000]),
    "backward_num_epochs": args.backward_num_epochs or tune.choice([1600, 2500]),
    "forward_batch_size": args.forward_batch_size or tune.choice([2**9]),
    "backward_batch_size": args.backward_batch_size or tune.choice([2**9]),
    "use_cache": args.use_cache,
    "kl_coeff": tune.loguniform(2**-3, 2**0),
    "kl_variance_coeff": tune.loguniform(2**-24, 2**0),
    "num_wavelens": 821,
    "prediction_iters": args.prediction_iters,
    "use_forward": args.use_forward,
    "load_forward_checkpoint": args.load_forward_checkpoint,
    "load_backward_checkpoint": args.load_backward_checkpoint,
}


# Make sure to pass in ``resources_per_trial`` using the ``get_tune_resources`` utility.
analysis = tune.run(
    main,
    metric="backward_val_loss",
    mode="min",
    config=config,
    num_samples=args.num_samples,
    resources_per_trial={"gpu": 1, "cpu": 64},
    name="tune_fwd_backward",
)

print("Best hyperparameters found were: ", analysis.best_config)
