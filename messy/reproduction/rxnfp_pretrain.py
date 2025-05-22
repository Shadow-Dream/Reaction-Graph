import numpy as np
import pandas as pd
from rxnfp.models import SmilesLanguageModelingModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 256,
  "initializer_range": 0.02,
  "intermediate_size": 512,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 2048,
  "model_type": "bert",
  "num_attention_heads": 4,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
}

vocab_path = 'vocab.txt'

args = {'config': config, 
        'vocab_path': vocab_path, 
        'train_batch_size': 32,
        'manual_seed': 42,
        "fp16": False,
        "num_train_epochs": 50,
        'max_seq_length': 2048,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'output_dir': 'weights/pretrain',
        'learning_rate': 1e-4
       }

model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args)

train_file = 'reaction_smiles_type_pretrain_train.txt'
eval_file = 'reaction_smiles_type_pretrain_eval.txt'
model.train_model(train_file=train_file, eval_file=eval_file)