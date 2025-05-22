import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn
import os
from rxnfp.models import SmilesClassificationModel

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dataframe_train = pd.read_csv("reaction_type_dataset_train.csv")
dataframe_test = pd.read_csv("reaction_type_dataset_test.csv")
dataframe_train = dataframe_train[["canonical_smiles","category"]]
dataframe_test = dataframe_test[["canonical_smiles","category"]]
dataframe_train.columns = ["text","labels"]
dataframe_test.columns = ["text","labels"]
dataframe_train = dataframe_train.sample(frac=1).reset_index(drop=True)
# final_train_df = dataframe_train.iloc[:16]
# eval_df = dataframe_train.iloc[16:24]
#前90%作为训练集
final_train_df = dataframe_train.iloc[:int(0.9*len(dataframe_train))]
#后10%作为验证集
eval_df = dataframe_train.iloc[int(0.9*len(dataframe_train)):]

# optional
model_args = {
    'num_train_epochs': 5, 
    'overwrite_output_dir': True,
    'learning_rate': 2e-5, 
    'gradient_accumulation_steps': 1,
    'regression': False, 
    "num_labels": 12, 
    "fp16": False,
    "evaluate_during_training": True, 
    'manual_seed': 42,
    "max_seq_length": 2048, 
    "train_batch_size": 8,
    "warmup_ratio": 0.00,
    'output_dir': 'weights/train', 
    'thread_count': 8,
    }

# model_path = pkg_resources.resource_filename("rxnfp", "")
model = SmilesClassificationModel("bert", None, num_labels=12, args=model_args, use_cuda=torch.cuda.is_available())
model.train_model(final_train_df, eval_df=eval_df, acc=sklearn.metrics.accuracy_score, mcc=sklearn.metrics.matthews_corrcoef)


# # optional
# train_model_path =  pkg_resources.resource_filename("rxnfp", "models/transformers/bert_class_1k_tpl")

# model = SmilesClassificationModel("bert", train_model_path, use_cuda=torch.cuda.is_available())

# # optional
# y_preds = model.predict(test_df.text.values)