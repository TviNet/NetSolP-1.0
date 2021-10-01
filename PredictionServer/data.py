# Copyright (c) 2021, Technical University of Denmark
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
import pickle
import torch


class FastaBatchedDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)
    
    def shuffle(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        return self.data_df["fasta"][idx], self.data_df["sid"][idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["fasta"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
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
        #print(len(raw_batch[0]), raw_batch[1], raw_batch[2])
        max_len = max(len(seq_str) for seq_str, _ in raw_batch)
        tokens = torch.empty((batch_size, max_len + int(self.alphabet.prepend_bos) + \
            int(self.alphabet.append_eos)), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        lengths = []
        strs = []
        for i, (seq_str, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor([self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
            tokens[i, int(self.alphabet.prepend_bos) : len(seq_str) + int(self.alphabet.prepend_bos)] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        
        non_pad_mask = ~tokens.eq(self.alphabet.padding_idx) &\
         ~tokens.eq(self.alphabet.cls_idx) &\
         ~tokens.eq(self.alphabet.eos_idx)# B, T

        return tokens, torch.tensor(lengths), non_pad_mask, labels

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