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
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

EMBEDDING_SIZE = 1280
EXTRACT_LAYER = 33

MODEL_NAME = "esm1b_t33_650M_UR50S"

class ESMFinetune(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)
        self.model = model
        self.clf_head = nn.Linear(EMBEDDING_SIZE, 1)
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        self.lr = 1e-5

    def forward(self, toks, lens, non_mask):
        # in lightning, forward defines the prediction/inference actions
        x = self.model(toks, repr_layers=[EXTRACT_LAYER])
        x = x["representations"][EXTRACT_LAYER]
        x_mean = (x * non_mask[:,:,None]).sum(1) / lens[:,None]
        x = self.clf_head(x_mean)
        return x.squeeze() 

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters()], 'lr': 3e-6},
            {"params": [p for n, p in self.clf_head.named_parameters()], 'lr': 2e-5},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        #self.unfreeze()
        x, l, n, y, _ = batch
        y_pred =  self.forward(x, l, n)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('train_loss_batch', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        #self.freeze()
        x, l, n, y, _ = batch
        y_pred =  self.forward(x, l, n)
        correct = ((y_pred>0) == y).sum()
        count = y.size(0)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('val_loss_batch', loss)
        return {'loss': loss, 'correct':correct, "count":count}
  
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        avg_acc = torch.tensor([x['correct'] for x in outputs]).sum() / torch.tensor([x['count'] for x in outputs]).sum()
        self.log('val_acc', avg_acc, prog_bar=True)