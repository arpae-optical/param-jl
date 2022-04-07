#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from ray import tune
from torch import Tensor

from data import DataModule
from model import ForwardBackwardModel
from nngraph import graph
from utils import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=None,
    help="Number of epochs.",
)

parser.add_argument(
    "--batch-size",
    "--bs",
    type=int,
    default=None,
)
parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="How many runs ",
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
    "--load-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained model. Useful for validation. Requires model to already be trained and saved.",
)
args = parser.parse_args()


def main(config: Config) -> None:
    config["save_dir"].mkdir(parents=True, exist_ok=True)
    run = wandb.init()

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=[
            WandbLogger(
                name="Laser params",
                save_dir="/data/alok/laser/wandb_logs",
                offline=False,
                project="Laser Forward",
                log_model=True,
            ),
        ],
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                dirpath="/data/alok/laser/weights",
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
        check_val_every_n_epoch=min(3, config["epochs"] - 1),
        gradient_clip_val=0.5,
        log_every_n_steps=min(3, config["epochs"] - 1),
    )

    data_module = DataModule(config)
    model = ForwardBackwardModel(config=config)
    if not config["load_checkpoint"]:
        trainer.fit(model=model, datamodule=data_module)
    trainer.test(
        model=model,
        ckpt_path=str(
            max(
                config["save_dir"] / "weights".glob("*.ckpt"),
                key=os.path.getctime,
            )
        ),
        datamodule=data_module,
    )

    for _ in range(config["prediction_iters"]):
        preds: List[Tensor] = trainer.predict(
            model=model,
            ckpt_path=str(
                max(
                    (config["save_dir"] / "weights").glob("*.ckpt"),
                    key=os.path.getctime,
                )
            ),
            datamodule=data_module,
            return_predictions=True,
        )
        save_filename = (
            config["save_dir"]
            / f"src/pred_iter_{config['prediction_iters']}_k1_variance_{config['kl_variance_coeff']}"
        )
        # TODO save artifact (maybe without upload if big)
        torch.save(preds, save_filename)
        artifact = wandb.Artifact(save_filename.name, type="result")
        artifact.add_file(save_filename)
        run.log_artifact(artifact)

    wandb.finish()


# The `or` idiom allows overriding values from the command line.
config: Config = {
    "lr": 1e-6,
    "epochs": args.epochs or tune.choice([4000]),
    "batch_size": args.batch_size or tune.choice([2**9]),
    "use_cache": args.use_cache,
    "kl_coeff": tune.loguniform(2**-1, 2**0),
    "kl_variance_coeff": tune.loguniform(2**-12, 2**0),
    "prediction_iters": args.prediction_iters,
    "load_checkpoint": args.load_checkpoint,
    "num_wavelens": 300,
    "save_dir": Path("/data/alok/laser"),
}


for i in range(1):
    # The `hasattr` lets us use Ray Tune just to provide hyperparameters.
    try:
        concrete_config: Config = Config(
            {k: (v.sample() if hasattr(v, "sample") else v) for k, v in config.items()}
        )
        main(concrete_config)
    except:
        try:
            # Don't want experiments bleeding into each other.
            wandb.finish()
        except:
            continue
        continue
