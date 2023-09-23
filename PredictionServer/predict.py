# Copyright (c) 2021, Technical University of Denmark
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
import onnxruntime
from data import *
import os
import pickle
import argparse
import pandas as pd
import time
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def get_preds_split(split_i, embed_dataloader, args, prediction_type, test_df):
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = args.NUM_THREADS
    opts.inter_op_num_threads = args.NUM_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    
    # Adjust session options
    if args.MODEL_TYPE == "Both":
        model_types = ["ESM12", "ESM1b"]
    else:
        model_types = [args.MODEL_TYPE]

    model_paths = [os.path.join(args.MODELS_PATH, f"{prediction_type}_{mt}_{split_i}_quantized.onnx") for mt in model_types]
    ort_sessions = [onnxruntime.InferenceSession(mp, sess_options=opts, providers=providers) for mp in model_paths]

    embed_dict = {}
    inputs_names = ort_sessions[0].get_inputs()
    for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
          #print(labels)
          ort_inputs = {inputs_names[0].name: toks.numpy(), inputs_names[1].name: lengths.numpy(), inputs_names[2].name: np_mask.numpy()}
          #print(ort_sessions[0].run(None, ort_inputs))
          ort_outs = [ort_session.run(None, ort_inputs)[0] for ort_session in ort_sessions]
          embed_dict[labels[0]] = sum(ort_outs) / len(ort_outs)

    pred_df = pd.DataFrame(embed_dict.items(), columns=["sid", "preds"])
    pred_df = test_df.merge(pred_df)
    #print(pred_df)
    return pred_df

def run_model_distilled(embed_dataloader, args, prediction_type, test_df):
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = args.NUM_THREADS
    opts.inter_op_num_threads = args.NUM_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    
    # Adjust session options
    model_paths = [os.path.join(args.MODELS_PATH,
      f"{prediction_type}_ESM1b_distilled_quantized.onnx")]
    ort_sessions = [onnxruntime.InferenceSession(mp, sess_options=opts) for mp in model_paths]

    embed_dict = {}
    inputs_names = ort_sessions[0].get_inputs()
    for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
          #print(labels)
          ort_inputs = {inputs_names[0].name: toks.numpy(), inputs_names[1].name: lengths.numpy(), inputs_names[2].name: np_mask.numpy()}
          #print(ort_sessions[0].run(None, ort_inputs))
          ort_outs = [ort_session.run(None, ort_inputs)[0] for ort_session in ort_sessions]
          embed_dict[labels[0]] = sum(ort_outs) / len(ort_outs)

    pred_df = pd.DataFrame(embed_dict.items(), columns=['sid', 'preds'])
    pred_df = test_df.merge(pred_df)
    #print(pred_df)
    return pred_df

def get_preds(args):
    fasta_dict = read_fasta(args.FASTA_PATH)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['sid', 'fasta'])
    test_df["fasta"] = test_df["fasta"].apply(lambda x: x[:1022])
    print(len(test_df))

    alphabet_path = os.path.join(args.MODELS_PATH,
      f"ESM12_alphabet.pkl")

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    #alphabet = Alphabet(proteinseq_toks)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)
    
    if "S" in args.PREDICTION_TYPE:
        print("Doing Solubility")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Solubility", test_df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            test_df[f"predicted_solubility_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        test_df["predicted_solubility"] = pd.Series(avg_pred)
    if "U" in args.PREDICTION_TYPE:
        print("Doing Usability")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Usability", test_df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            test_df[f"predicted_usability_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        test_df["predicted_usability"] = pd.Series(avg_pred)

    test_df.to_csv(args.OUTPUT_PATH, index=False)

def get_preds_distilled(args):
    fasta_dict = read_fasta(args.FASTA_PATH)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['sid', 'fasta'])
    test_df["fasta"] = test_df["fasta"].apply(lambda x: x[:1022])
    print(len(test_df))

    alphabet_path = os.path.join(args.MODELS_PATH,
      f"ESM12_alphabet.pkl")

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    #alphabet = Alphabet(proteinseq_toks)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)
    
    if "S" in args.PREDICTION_TYPE:
        print("Doing Solubility")
        pred_df = run_model_distilled(embed_dataloader, args, "Solubility", test_df)
        preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
        test_df[f"predicted_solubility"] = preds_i
    if "U" in args.PREDICTION_TYPE:
        print("Doing Usability")
        pred_df = run_model_distilled(embed_dataloader, args, "Usability", test_df)
        preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
        test_df[f"predicted_usability"] = preds_i

    test_df.to_csv(args.OUTPUT_PATH, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FASTA_PATH", type=str, help="Input protein sequences in the FASTA format"
    )
    parser.add_argument(
        "--OUTPUT_PATH", type=str, help="Output location of predictions as CSV"
    )
    parser.add_argument(
        "--MODELS_PATH", default="./models/", type=str, help="Location of models"
    )
    parser.add_argument(
        "--NUM_THREADS",
        default=os.cpu_count(),
        type=int,
        help="Number of threads to use. (Use more for faster results)"
    )
    parser.add_argument(
        "--MODEL_TYPE", 
        default="ESM1b",
        choices=['ESM12', 'ESM1b', 'Both', 'Distilled'],
        type=str,
        help="Model to use. ESM1b is better but much slower. Both option averages the prediction"
    )
    parser.add_argument(
        "--PREDICTION_TYPE",
        default="S",
        choices=['S', 'U', 'SU'],
        type=str,
        help="Either Solubility(S), Usability(U) or Both"
    )
    args = parser.parse_args()
    t1 = time.time()
    if args.MODEL_TYPE == "Distilled":
        get_preds_distilled(args)
    else:
        get_preds(args)
    print(f"Finished prediction in {time.time()-t1}s")