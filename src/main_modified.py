from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger

from backwards import BackwardDataModule, BackwardModel
from forwards import ForwardDataModule, ForwardModel

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
    "--use-fwd",
    type=eval,
    choices=[True, False],
    default=True,
    help="Whether to use a forward model at all",
)
parser.add_argument(
    "--load-checkpoint",
    type=eval,
    choices=[True, False],
    default=False,
    help="Load trained model. Useful for validation. Requires model to already be trained and saved.",
)
args = parser.parse_args()


forward_trainer = pl.Trainer(
    max_epochs=args.forward_num_epochs,
    logger=[
        TestTubeLogger(
            save_dir="test_tube_logs/forward", name="Forward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="forward/train/loss",
            dirpath="weights/forward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=10,
    gradient_clip_val=0.5,
    log_every_n_steps=10,
)


backward_trainer = pl.Trainer(
    max_epochs=args.backward_num_epochs,
    logger=[
        TestTubeLogger(
            save_dir="test_tube_logs/backward", name="Backward", create_git_tag=False
        ),
    ],
    callbacks=[
        ModelCheckpoint(
            monitor="backward/train/loss",
            dirpath="weights/backward",
            save_top_k=1,
            mode="min",
        ),
    ],
    gpus=torch.cuda.device_count(),
    precision=32,
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=10,
    check_val_every_n_epoch=10,
    gradient_clip_val=0.5,
    log_every_n_steps=30,
)

# TODO: load checkpoint for both forward and back
if args.use_fwd:
    forward_model = ForwardModel()
    forward_data_module = ForwardDataModule(
        batch_size=args.forward_batch_size,
        use_cache=args.use_cache,
    )
    if not args.load_checkpoint:
        forward_trainer.fit(model=forward_model, datamodule=forward_data_module)

    forward_trainer.test(
        model=forward_model,
        ckpt_path="best",
        datamodule=forward_data_module,
    )
    backward_model = BackwardModel(forward_model=forward_model)
else:
    backward_model = BackwardModel(forward_model=None)

backward_data_module = BackwardDataModule(
    batch_size=args.backward_batch_size,
    use_cache=args.use_cache,
)
if not args.load_checkpoint:
    backward_trainer.fit(model=backward_model, datamodule=backward_data_module)
backward_trainer.test(
    model=backward_model,
    ckpt_path="best",
    datamodule=backward_data_module,
)

import data
from utils import Stage, rmse, split
from pathlib import Path
real_laser, real_emissivity = data.get_data()

predicted_laser = backward_model(real_emissivity)
predicted_emissivity = forward_model(predicted_laser)




torch.save(real_laser, Path("real_laser.pt"))
torch.save(real_emissivity, Path("real_emissivity.pt"))
torch.save(predicted_laser, Path("predicted_laser.pt"))
torch.save(predicted_emissivity, Path("backwards_val_output.pt"))