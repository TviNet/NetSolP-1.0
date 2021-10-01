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
from models import *
from dataloader import *
pl.trainer.seed_everything(0)

EMBEDDING_SIZE = 768#1280
EXTRACT_LAYER = 12#33

MAX_LENGTH = 511

# TARGET_BATCH_SIZE = 64
# BATCH_SIZE = 8
NUM_GPUS = 1 # 2
GRAD_ACCM = 1 #TARGET_BATCH_SIZE // (BATCH_SIZE * NUM_GPUS)
MAX_TOKENS_PER_BATCH = 4096
NUM_STEPS = 1000
MODEL_NAME = "esm1_t12_85M_UR50S"#"esm1b_t33_650M_UR50S"
path = 'models/'



from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, matthews_corrcoef
import numpy as np

def camsol_test(split_i, clf):
    test_df = pd.read_csv("../Datasets/Camsol/CamsolMut_cleaned.csv")
    print(len(test_df))
    newalphabet = NewAlphabet(alphabet)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(MAX_TOKENS_PER_BATCH, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=newalphabet.get_batch_converter(), batch_sampler=embed_batches)

    embed_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
          x = torch.sigmoid(clf(toks.to("cuda"), lengths.to("cuda"), np_mask.to("cuda"))).cpu().numpy()
          for j in range(len(labels)):
              if len(labels) == 1:
                embed_dict[labels[j]] = x
              else:
                embed_dict[labels[j]] = x[j]

    pred_df = pd.DataFrame(embed_dict.items(), columns=['sid', 'preds'])
    test_df = test_df.merge(pred_df)
    return test_df

preds_avg = []

for split_i in range(5):
    path = f"models/{split_i}PSISplit.ckpt"
    clf = ESMFinetune.load_from_checkpoint(path)
    clf.eval().cuda()
    pred_df = camsol_test(split_i, clf)
    pred_df["pred_effect"] = 0
    preds_avg.append(pred_df["preds"])
    for i in range(len(pred_df)):
        if pred_df["Mutation"][i] != "WT":
          pred_df.loc[i, "pred_effect"] = np.sign(pred_df.loc[i, "preds"] - pred_df[(pred_df["Protein Name"] == pred_df["Protein Name"][i])& (pred_df["Mutation"] == "WT")].preds.item())
    test_set = pred_df[~pred_df["Observed Effect"].isna()].reset_index(drop=True)
    test_set["ObservedPred"] = test_set["Observed Effect"].apply(lambda x: +1 if x == "+" else -1)
    print(f"Fold {split_i}:", (test_set["ObservedPred"] == test_set["pred_effect"]).mean())

pred_df = pd.read_csv("../Datasets/Camsol/CamsolMut_cleaned.csv")
pred_df["pred_effect"] = 0
pred_df["preds"] = sum(preds_avg) / 5
for i in range(len(pred_df)):
    if pred_df["Mutation"][i] != "WT":
      pred_df.loc[i, "pred_effect"] = np.sign(pred_df.loc[i, "preds"] - pred_df[(pred_df["Protein Name"] == pred_df["Protein Name"][i])& (pred_df["Mutation"] == "WT")].preds.item())
test_set = pred_df[~pred_df["Observed Effect"].isna()].reset_index(drop=True)
test_set["ObservedPred"] = test_set["Observed Effect"].apply(lambda x: +1 if x == "+" else -1)
print((test_set["ObservedPred"] == test_set["pred_effect"]).mean())
print(test_set[(test_set["ObservedPred"] != test_set["pred_effect"])])