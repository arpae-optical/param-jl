#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger

from backwards import BackwardDataModule, BackwardModel
from forwards import ForwardDataModule, ForwardModel

parser = argparse.ArgumentParser()
parser.add_argument("--forward-num-epochs", "-bn", type=int, default=2_000)
parser.add_argument("--backward-num-epochs", "-fn", type=int, default=2_000)
parser.add_argument("--forward-batch-size", "--fb", type=int, default=256)
parser.add_argument("--backward-batch-size", "--bb", type=int, default=256)
parser.add_argument("--use-cache", type=eval, choices=[True, False], default=True)
args = parser.parse_args()


forward_trainer = pl.Trainer(
    max_epochs=args.forward_num_epochs,
    logger=[
        WandbLogger(
            name="Forward laser params",
            save_dir="wandb_logs/forward",
            offline=False,
            project="Laser",
            log_model=True,
        ),
        TestTubeLogger(
            save_dir="test_tube_logs/forward", name="Forward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="weights/forward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
    gradient_clip_val=0.5,
)


backward_trainer = pl.Trainer(
    max_epochs=args.backward_num_epochs,
    logger=[
        WandbLogger(
            name="Backward laser params",
            save_dir="wandb_logs/backward",
            offline=False,
            project="Laser",
            log_model=True,
        ),
        TestTubeLogger(
            save_dir="test_tube_logs/backward", name="Backward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="weights/backward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
)

forward_model = ForwardModel()
forward_data_module = ForwardDataModule(
    batch_size=args.forward_batch_size,
    use_cache=args.use_cache,
)
forward_trainer.fit(forward_model, datamodule=forward_data_module)

backward_model = BackwardModel(forward_model=forward_model)
backward_data_module = BackwardDataModule(
    batch_size=args.backward_batch_size,
    use_cache=args.use_cache,
)
backward_trainer.fit(backward_model, datamodule=backward_data_module)
