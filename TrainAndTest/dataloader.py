# Copyright (c) 2021, Technical University of Denmark
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
import re
import torch
import random
from esm import Alphabet, FastaBatchedDataset, pretrained
import pickle
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

MAX_LENGTH = 510
TARGET_BATCH_SIZE = 64
BATCH_SIZE = 4
NUM_GPUS = 2
GRAD_ACCM = TARGET_BATCH_SIZE // (BATCH_SIZE * NUM_GPUS)
MAX_TOKENS_PER_BATCH = 4096
NUM_STEPS = 1000

class FastaDataset(object):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df["fasta"][idx],self.data_df["solubility"][idx],self.data_df["sid"][idx]

class FastaBatchedDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)
    
    def shuffle(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        return self.data_df["fasta"][idx], self.data_df["solubility"][idx], self.data_df["sid"][idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["fasta"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0 or len(buf) == 1:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch):
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches



class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        tokens = torch.empty((batch_size, MAX_LENGTH + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        lengths = []
        strs = []
        targets = torch.zeros((batch_size,), dtype=torch.float32)
        for i, (seq_str, target, label) in enumerate(raw_batch):
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            targets[i] = target
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor([self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
            tokens[i, int(self.alphabet.prepend_bos) : len(seq_str) + int(self.alphabet.prepend_bos)] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        non_pad_mask = ~tokens.eq(self.alphabet.padding_idx) &\
         ~tokens.eq(self.alphabet.cls_idx) &\
         ~tokens.eq(self.alphabet.eos_idx)# B, T
            
        return tokens, torch.tensor(lengths), non_pad_mask, targets, labels # dct_mat, idct_mat, 

class Alphabet(object):
    prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
    append_toks = ("<cls>", "<mask>", "<sep>")
    prepend_bos = True
    append_eos = True

    def __init__(self, standard_toks):
        self.standard_toks = list(standard_toks)

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return {"toks": self.toks}

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_dict(cls, d):
        return cls(standard_toks=d["toks"])

# class RobertaAlphabet(Alphabet):
#     prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
#     append_toks = ("<mask>",)
#     prepend_bos = True
#     append_eos = True

class NewAlphabet(Alphabet):
    def __init__(self, alphabet):
        self.alphabet = alphabet
    def get_batch_converter(self):
        return BatchConverter(self.alphabet)



def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    with open(fastafile, 'r') as f:
        content = [l.strip() for l in f.readlines()]

    res = {}
    seq, seq_id = '', None
    for line in content:
        if line.startswith('>'):
            
            if len(seq) > 0:
                res[seq_id] = seq
            
            seq_id = line.replace('>', '')
            seq = ''
        else:
            seq += line
    res[seq_id] = seq
    return res


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
#print(len(fasta_df))

data_df = labels_df.merge(fasta_df)

clusters_df = pd.read_csv(CLUSTERS_FILE)
clusters_df.columns = ["sid","priority","label-val","between_connectivity","cluster"]

data_df = data_df.merge(clusters_df)
with open("ESM12_alphabet.pkl", "rb") as f:
    alphabet = pickle.load(f)
newalphabet = NewAlphabet(alphabet)

def get_next_split(i):

    train_df = data_df[data_df.cluster != i].reset_index(drop=True)

    train_df["fasta"] = train_df["fasta"].apply(lambda x: x[:MAX_LENGTH])
    #train_df = train_df[train_df["lengths"] <= MAX_LENGTH].reset_index(drop=True)
    
    X, y = np.stack(train_df["sid"].to_numpy()), np.stack(train_df["solubility"].to_numpy())
    sss_tt = StratifiedShuffleSplit(n_splits=1, test_size=512, random_state=0)
    
    (split_train_idx, split_val_idx) = next(sss_tt.split(X, y))
    split_train_df =  train_df.iloc[split_train_idx].reset_index(drop=True)
    split_val_df = train_df.iloc[split_val_idx].reset_index(drop=True)

    #NO_OF_STEPS = len(split_train_df) // BATCH_SIZE
    print(len(split_train_df))

    train_dataset = FastaDataset(split_train_df)
    train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      collate_fn=newalphabet.get_batch_converter(),
      shuffle=True,
      batch_size=BATCH_SIZE,
      num_workers=4,
      #pin_memory=True,
      drop_last=True)

    val_dataset = FastaDataset(split_val_df)
    val_dataloader = torch.utils.data.DataLoader(
      val_dataset,
      collate_fn=newalphabet.get_batch_converter(),
      #num_workers=4,
      shuffle=False,
        batch_size=BATCH_SIZE)
    
    return train_dataloader, val_dataloader
