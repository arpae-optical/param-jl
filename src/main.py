#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor

from backwards import BackwardModel
from data import BackwardDataModule, ForwardDataModule, StepTestDataModule
from forwards import ForwardModel
from nngraph import graph

parser = argparse.ArgumentParser()
parser.add_argument(
    "--forward-num-epochs",
    "--fe",
    type=int,
    default=5_000,
    help="Number of epochs for forward model",
)
parser.add_argument(
    "--backward-num-epochs",
    "--be",
    type=int,
    default=5_000,
    help="Number of epochs for backward model",
)
parser.add_argument(
    "--forward-batch-size",
    "--fbs",
    type=int,
    default=512,
    help="Batch size for forward model",
)
parser.add_argument(
    "--backward-batch-size",
    "--bbs",
    type=int,
    default=512,
    help="Batch size for backward model",
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
    help="Load trained model. Useful for validation. Requires model to already be trained and saved.",
)
parser.add_argument(
    "--load-backward-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained model. Useful for validation. Requires model to already be trained and saved.",
)
args = parser.parse_args()

fwd_checkpoint_cb = ModelCheckpoint(
    monitor="forward/train/loss",
    dirpath="weights/forward",
    save_top_k=1,
    mode="min",
)

backward_checkpoint_cb = ModelCheckpoint(
    monitor="backward/train/loss",
    dirpath="weights/backward",
    save_top_k=1,
    mode="min",
)

forward_trainer = pl.Trainer(
    max_epochs=args.forward_num_epochs,
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
        fwd_checkpoint_cb,
        pl.callbacks.progress.TQDMProgressBar(refresh_rate=100),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    check_val_every_n_epoch=10,
    gradient_clip_val=0.5,
    log_every_n_steps=min(10, args.forward_num_epochs),
)


backward_trainer = pl.Trainer(
    max_epochs=args.backward_num_epochs,
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
        backward_checkpoint_cb,
        pl.callbacks.progress.TQDMProgressBar(refresh_rate=10),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    check_val_every_n_epoch=10,
    gradient_clip_val=0.5,
    log_every_n_steps=min(30, args.backward_num_epochs),
)

forward_data_module = ForwardDataModule(
    batch_size=args.forward_batch_size,
    use_cache=args.use_cache,
)
backward_data_module = BackwardDataModule(
    batch_size=args.backward_batch_size,
    use_cache=args.use_cache,
)
step_data_module = StepTestDataModule()

# "/home/alok/param_jl/ARPAE/weights/forward/epoch=4649-step=223199.ckpt"
# "/home/alok/param_jl/ARPAE/weights/forward/epoch=4979-step=239039.ckpt"
# "/home/alok/param_jl/ARPAE/weights/backward/epoch=4339-step=208319.ckpt"
# "/home/alok/param_jl/ARPAE/weights/backward/epoch=2159-step=103679.ckpt"

# TODO: load checkpoint for both forward and back
if args.use_forward:
    forward_model = ForwardModel()
    if not args.load_forward_checkpoint:
        forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

    # XXX needs to be cast to str to avoid a TypeError in the wandb logger.
    fwd_checkpoint_cb.best_model_path = str(
        max(
            Path("/home/alok/param_jl/ARPAE/weights/forward").glob("*.ckpt"),
            key=os.path.getctime,
        )
    )
    forward_trainer.test(
        model=forward_model,
        ckpt_path=str(fwd_checkpoint_cb.best_model_path),
        datamodule=forward_data_module,
    )
    backward_model = BackwardModel(forward_model=forward_model)
else:
    backward_model = BackwardModel(forward_model=None)

if not args.load_backward_checkpoint:
    backward_trainer.fit(model=backward_model, datamodule=backward_data_module)
# XXX needs to be cast to str to avoid a TypeError in the wandb logger.
backward_checkpoint_cb.best_model_path = str(
    max(
        Path("/home/alok/param_jl/ARPAE/weights/backward").glob("*.ckpt"),
        key=os.path.getctime,
    )
)
backward_trainer.test(
    model=backward_model,
    ckpt_path=str(backward_checkpoint_cb.best_model_path),
    datamodule=backward_data_module,
)

preds: List[Tensor] = backward_trainer.predict(
    model=backward_model,
    ckpt_path=backward_checkpoint_cb.best_model_path,
    datamodule=backward_data_module,
    return_predictions=True,
)
torch.save(preds, f"src/preds_i_validation/preds_0_validation")

# residuals is param vs emiss residuals, preds vs true is the graph of one true emiss vs 20 vae predicitons
# index str precedes all saved image files (f'{index_str}_graph_buckets.png' and f'{index_str}_vs_training_best_{i_run_index}.png')
graph(residualsflag = True, predsvstrueflag = True, index_str = "default")