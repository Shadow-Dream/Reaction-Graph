import pandas as pd
import json
from tqdm import tqdm
import argparse
from rxnmapper import RXNMapper
import os
import re
dataset = pd.DataFrame()
for i in range(20):
    dataset = pd.concat([dataset,pd.read_csv(f"condition/stage8/dataset_{i}.csv")])
dataset.fillna("",inplace=True)
#drop掉mapped_smiles列為空的行
dataset = dataset[dataset["mapped_smiles"]!=""]
dataset.to_csv("condition/stage9/dataset.csv",index=False)