from databases.CRM_dataset import CRMDataset
from models.CRM_model import CRMModel
from argparse import ArgumentParser
import numpy as np
import random
import torch
import os

parser = ArgumentParser(description='hyperparameters')
parser.add_argument('--seed', type=float, default=123)
parser.add_argument('--device', type=str, default="0")
parser.add_argument("--model_dir",type=str, default="weights/CRM")

parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-10)
parser.add_argument("--learning_rate_schedule", type=str,default="min") 
parser.add_argument("--learning_rate_factor", type=float,default=0.1)
parser.add_argument("--learning_rate_patience", type=int,default=5)
parser.add_argument("--min_learning_rate", type=float,default=1e-8)
parser.add_argument("--learning_rate_verbose", type=bool,default=True)

parser.add_argument("--dim_preprocess", type=int,default = 3000)
parser.add_argument("--dim_hidden", type=int,default = 300)
parser.add_argument("--dim_message", type=int,default = 200)
parser.add_argument("--dropout", type=float,default = 0.5)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
dataset_train = CRMDataset("/xxx/chem/USPTO_Condition",type="train",merge=True)
dataset_test = CRMDataset("/xxx/chem/USPTO_Condition",type="test",merge=True)
dataset_val = CRMDataset("/xxx/chem/USPTO_Condition",type="val",merge=True)
model = CRMModel(dataset_train=dataset_train,dataset_test = dataset_test,dataset_val = dataset_val,
                model_dir = args.model_dir,
                parameters_for_model={
                    "dim_preprocess":args.dim_preprocess,
                    "dim_hidden":args.dim_hidden,
                    "dim_message":args.dim_message,
                    "dropout":args.dropout,
                },
                parameters_for_optimizer = {
                    "lr":args.learning_rate, 
                    "weight_decay":args.weight_decay},
                parameters_for_scheduler={
                    "mode":args.learning_rate_schedule, 
                    "factor":args.learning_rate_factor, 
                    "patience":args.learning_rate_patience, 
                    "min_lr":args.min_learning_rate, 
                    "verbose":args.learning_rate_verbose})
model.train(save_delta = 5)