# Copyright (c) 2021, Technical University of Denmark
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
from esm import Alphabet, FastaBatchedDataset, pretrained
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pickle
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader import *
from models import *
pl.trainer.seed_everything(0)

path = 'models/'


def train_model(idx): 
    train_dataloader, val_dataloader = get_next_split(idx)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=path,
        filename= f"{idx}" + 'PSISplit-{epoch:02d}-{val_loss:.2f}',
        period=1,
        save_top_k=1,
        save_last=False
    )

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=5, 
                        check_val_every_n_epoch=1, 
                        default_root_dir=path + f"{idx}",
                        callbacks=[checkpoint_callback],
                        #precision=16,
                        progress_bar_refresh_rate=NUM_STEPS,
                        accumulate_grad_batches=GRAD_ACCM,
                        gpus=1)
                        #accelerator="ddp",
                        #plugins="ddp_sharded")
    clf = ESMFinetune()
    print(f"Training clf {idx}")
    trainer.fit(clf, train_dataloader, val_dataloader)
    #trainer.save_checkpoint(path + f"{idx}.ckpt")

print("Training model....")
for i in range(5):
  train_model(i)