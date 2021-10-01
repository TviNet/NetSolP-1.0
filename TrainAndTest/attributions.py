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
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import matplotlib.pyplot as plt
from models import *
from dataloader import *
pl.trainer.seed_everything(0)

EMBEDDING_SIZE = 1280
EXTRACT_LAYER = 33

MAX_LENGTH = 510

NUM_GPUS = 1 
MODEL_NAME = "esm1b_t33_650M_UR50S"

import pickle
def predict(toks, lengths, np_mask):
    return clf(toks, lengths, np_mask)

def custom_forward(toks, lengths, np_mask):
    preds = predict(toks, lengths, np_mask)
    return torch.sigmoid(preds)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

for split_i in range(5):
    print(split_i)
    path = f"models/{split_i}PSISplit.ckpt"
    clf = ESMFinetune.load_from_checkpoint(path).cuda()
    clf.zero_grad()

    lig = LayerIntegratedGradients(custom_forward, clf.model.embed_tokens)
    data_df = pd.read_csv("../Datasets/NESG/NESG_testset.csv")

    newalphabet = NewAlphabet(alphabet)
    embed_dataset = FastaBatchedDataset(data_df)
    embed_batches = embed_dataset.get_batch_indices(2048, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=newalphabet.get_batch_converter(), batch_sampler=embed_batches)

    score_vises_dict = {}
    attribution_dict = {}
    pred_dict = {}
    for (toks, lengths, np_mask, labels) in embed_dataloader:
        baseline_toks = torch.empty((toks.size(0), toks.size(1)), dtype=torch.int64)
        baseline_toks.fill_(newalphabet.alphabet.cls_idx)
        baseline_toks[:, 0] = newalphabet.alphabet.cls_idx
        attributions, delta = lig.attribute(inputs=toks.to("cuda"),
                                        baselines=baseline_toks.to("cuda"),
                                        n_steps=50,
                                        additional_forward_args=(lengths.to("cuda"), np_mask.to("cuda")),
                                        internal_batch_size=8,
                                        return_convergence_delta=True)
        
        preds = custom_forward(toks.to("cuda"),lengths.to("cuda"), np_mask.to("cuda"))

        for i in range(preds.shape[0]):
            attributions_sum = summarize_attributions(attributions[i])
            attribution_dict[labels[i]] = attributions_sum.cpu().numpy()[1:1+lengths[i]]
            pred_dict[labels[i]] = preds[i].cpu().detach().numpy()

    with open(f"esm1b_nesg_{split_i}_attrs.pkl", "wb") as f:
        pickle.dump({"attributions": attribution_dict, "preds": pred_dict}, f)