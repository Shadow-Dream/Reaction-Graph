from databases.parrot_dataset import ParrotDataset
from models.parrot_model import ParrotModel
from argparse import ArgumentParser
import numpy as np
import torch
import random
import os

parser = ArgumentParser(description='hyperparameters')
parser.add_argument('--seed', type=float,default=123)
parser.add_argument('--device', type=str,default="0")
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
dataset_val = ParrotDataset("chem/USPTO_Condition","val",length=68076//8 + 1,merge=True)
dataset_test = ParrotDataset("chem/USPTO_Condition","test",length=68076//8 + 1,merge=True)
model = ParrotModel(dataset_val= dataset_val,dataset_test = dataset_test,model_dir="weights/Parrot")

model.load("enhance")
accuracy = model.validate()
topk_row="topk        "
for topk in accuracy["overall"]:
    topk_row += " "  +(str(topk) + "      ")[:6]
print(topk_row)
for key in accuracy:
    accuracy_row = (key + "            ")[:12]
    for topk in accuracy[key]:
        accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
    print(accuracy_row)