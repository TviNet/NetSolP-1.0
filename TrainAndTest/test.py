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

EMBEDDING_SIZE = 1280
EXTRACT_LAYER = 33

MAX_LENGTH = 510

# TARGET_BATCH_SIZE = 64
# BATCH_SIZE = 8
NUM_GPUS = 1 # 2
GRAD_ACCM = 1 #TARGET_BATCH_SIZE // (BATCH_SIZE * NUM_GPUS)
MAX_TOKENS_PER_BATCH = 4096
NUM_STEPS = 1000
MODEL_NAME = "esm1b_t33_650M_UR50S"
path = 'models/'


from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, matthews_corrcoef
import numpy as np

def evaluate_split(split_i, test_df):
    net1_probs_test = np.stack(test_df["preds"].to_numpy())
    y_test = np.stack(test_df["solubility"].to_numpy())

    net1_preds_test = net1_probs_test>0.5

    acc = accuracy_score(y_test, net1_preds_test)
    pre = precision_score(y_test, net1_preds_test)
    mcc = matthews_corrcoef(y_test, net1_preds_test)
    auc = roc_auc_score(y_test, net1_probs_test)

    print(f"Fold{i}- Acc: {acc:.3f}, Pre: {pre:.3f}, MCC: {mcc:.3f}, AUC: {auc:.3f}\n")
    return acc, pre, mcc, auc
    

def psi_test_split(split_i, clf):
    FASTA_FILE = "../Datasets/PSI_Biology/pET_full_without_his_tag.fa"
    LABELS_FILE = "../Datasets/PSI_Biology/class.txt"
    CLUSTERS_FILE = "../Datasets/PSI_Biology/psi_biology_nesg_partitioning_wl_th025_amT.csv"

    labels_df = pd.read_csv(LABELS_FILE, delimiter="\t")
    labels_df.columns = ["sid", "solubility"]
    labels_df.solubility = labels_df.solubility -1

    fasta_dict = read_fasta(FASTA_FILE)
    fasta_df = pd.DataFrame(fasta_dict.items(), columns=['Accession', 'fasta'])
    fasta_df["sid"] = fasta_df.Accession.apply(lambda x: x.split("_")[0])
    #fasta_df.sid = fasta_df.sid.astype('int64') 
    print(len(fasta_df))

    data_df = labels_df.merge(fasta_df)

    clusters_df = pd.read_csv(CLUSTERS_FILE)
    clusters_df.columns = ["sid","priority","label-val","between_connectivity","cluster"]

    data_df = data_df.merge(clusters_df)
    test_df = data_df[data_df.cluster == split_i].reset_index(drop=True)
    print(len(test_df))
    newalphabet = NewAlphabet(alphabet)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(MAX_TOKENS_PER_BATCH, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=newalphabet.get_batch_converter(), batch_sampler=embed_batches)

    embed_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, targets, labels) in enumerate(embed_dataloader):
          x = torch.sigmoid(clf(toks.to("cuda"), lengths.to("cuda"), np_mask.to("cuda"))).cpu().numpy()
          for j in range(len(labels)):
              if len(labels) == 1:
                embed_dict[labels[j]] = x
              else:
                embed_dict[labels[j]] = x[j]

    pred_df = pd.DataFrame(embed_dict.items(), columns=['sid', 'preds'])
    test_df = test_df.merge(pred_df)
    evaluate_split(split_i, test_df)
    return test_df

def psi_nesg_test(split_i, clf):
    test_df = pd.read_csv("../Datasets/NESG/NESG_testset.csv")
    print(len(test_df))
    newalphabet = NewAlphabet(alphabet)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(MAX_TOKENS_PER_BATCH, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=newalphabet.get_batch_converter(), batch_sampler=embed_batches)

    embed_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, targets, labels) in enumerate(embed_dataloader):
          x = torch.sigmoid(clf(toks.to("cuda"), lengths.to("cuda"), np_mask.to("cuda"))).cpu().numpy()
          for j in range(len(labels)):
              if len(labels) == 1:
                embed_dict[labels[j]] = x
              else:
                embed_dict[labels[j]] = x[j]

    pred_df = pd.DataFrame(embed_dict.items(), columns=['sid', 'preds'])
    test_df = test_df.merge(pred_df)
    return evaluate_split(split_i, test_df)

accs = []
pres = []
mccs = []
aucs = []

for i in range(5):
    path = f"models/{i}PSISplit.ckpt"
    clf = ESMFinetune.load_from_checkpoint(path)
    clf.eval().cuda()
    acc, pre, mcc, auc = psi_test_split(i, clf)
    accs.append(acc)
    pres.append(pre)
    mccs.append(mcc)
    aucs.append(auc)

print(f"{round(np.array(accs).mean(), 2)} + {round(np.array(accs).std(), 2)}" + " & "
      f"{round(np.array(pres).mean(), 2)} + {round(np.array(pres).std(), 2)}" + " & "
      f"{round(np.array(mccs).mean(), 2)} + {round(np.array(mccs).std(), 2)}" + " & "
      f"{round(np.array(aucs).mean(), 2)} + {round(np.array(aucs).std(), 2)}" + " \\\\ ")