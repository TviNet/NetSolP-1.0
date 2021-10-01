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
pl.trainer.seed_everything(0)

EMBEDDING_SIZE = 1280
EXTRACT_LAYER = 33


MAX_TOKENS_PER_BATCH = 4096
NUM_STEPS = 1000
MODEL_NAME = "esm1b_t33_650M_UR50S"
path = 'models/'

PATH_MODEL_PREFIX = "PSIBiology/models_finetuning/"
PATH_ONNX_MODEL_PREFIX = 'PSIBiology/ONNX_models/'
PATH_ONNX_QUANTIZED_MODEL_PREFIX = "PSIBiology/ONNX_models/"

print("Loading model....")
model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)

###################################################################################################################################################
import re
import torch
import random

###################################################################################################################################################
# title finetuning model mean
class ESMFinetune(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)
        self.model = model
        self.clf_head = nn.Linear(EMBEDDING_SIZE, 1)
        self.lr = 1e-5

    def forward(self, toks, lens, non_mask):
        # in lightning, forward defines the prediction/inference actions
        x = self.model(toks, repr_layers=[EXTRACT_LAYER])
        x = x["representations"][EXTRACT_LAYER]
        x_mean = (x * non_mask[:,:,None]).sum(1) / lens[:,None]
        x = self.clf_head(x_mean)
        return x.squeeze()
###################################################################################################################################################
with open(PATH_ONNX_MODEL_PREFIX + "ESM12_alphabet.pkl", "wb") as f:
    pickle.dump(alphabet, f)

from onnxruntime.quantization import quantize, QuantizationMode,quantize_dynamic, quantize_static
import onnx
# Export the model
with torch.no_grad():
    for i in range(5):
            path = PATH_MODEL_PREFIX + f"{i}PSISplit.ckpt"
            clf = ESMFinetune.load_from_checkpoint(path).eval()
            toks = 5*torch.ones((1,1024), dtype=torch.int64)
            lengths = torch.ones((1,), dtype=torch.int64)
            np_mask = toks.eq(5)
            
            #toks, lengths, np_mask, targets, labels = next(iter(embed_dataloader))
            # output = clf(toks, lengths, np_mask)
            # print(output)
            
            torch.onnx.export(clf, # model being run
                              (toks, lengths, np_mask),  # model input (or a tuple for multiple inputs)
                              PATH_ONNX_MODEL_PREFIX,   # where to save the model (can be a file or file-like object)
                              export_params=True,        # store the trained parameter weights inside the model file
                              opset_version=13,          # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names = ['tokens', 'lengths', 'non_pad_mask'],   # the model's input names
                              output_names = ['output'], # the model's output names
                              dynamic_axes={'tokens' : {0 : 'batch_size', 1: 'seq_len'},    # variable lenght axes
                                            'non_pad_mask' : {0 : 'batch_size', 1: 'seq_len'},
                                            'lengths' : {0 : 'batch_size'},
                                            'output' : {0 : 'batch_size'}})


            onnx_model = onnx.load(PATH_ONNX_MODEL_PREFIX + f"ESM1b_{i}.onnx")
            quantized_model = quantize(
                    model=onnx_model,
                    quantization_mode=QuantizationMode.IntegerOps,
                    force_fusions=True,
                    symmetric_weight=True,
            )
            onnx.save_model(quantized_model, PATH_ONNX_QUANTIZED_MODEL_PREFIX + f"ESM1b_{i}_quantized.onnx")

