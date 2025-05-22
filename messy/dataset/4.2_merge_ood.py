import argparse
import pandas as pd
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("params")
parser.add_argument("--ood_dir",type=str,default="stage3.2")
args = parser.parse_args()

ood_dir = args.ood_dir
file_paths = [os.path.join(ood_dir,file_name) for file_name in os.listdir(ood_dir) if file_name.endswith(".csv")]
datasets = [pd.read_csv(file_path) for file_path in tqdm(file_paths)]
#更改agent1列为reagent1
for dataset in datasets:
    dataset.fillna("",inplace=True)
    dataset.rename(columns={"agent1":"reagent1"},inplace=True)
    dataset.rename(columns={"agent2":"reagent2"},inplace=True)
    #去除catalyst1,solvent1,solvent2,reagent1,reagent2列全为""的行
    dataset = dataset[~((dataset["catalyst1"]=="") & (dataset["solvent1"]=="") & (dataset["solvent2"]=="") & (dataset["reagent1"]=="") & (dataset["reagent2"]==""))]
dataset = pd.concat(datasets)
dataset = dataset.drop_duplicates()
dataset.to_csv("stage4.2/dataset.csv",index=False)