import argparse
import pandas as pd
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("params")
parser.add_argument("--stage5_dir",type=str,default="stage5")
args = parser.parse_args()

stage5_dir = args.stage5_dir
file_paths = [os.path.join(stage5_dir,file_name) for file_name in os.listdir(stage5_dir) if file_name.endswith(".csv")]
datasets = [pd.read_csv(file_path) for file_path in tqdm(file_paths)]
dataset = pd.concat(datasets)

dataset = dataset.drop_duplicates()

dataset.to_csv("stage6/dataset.csv",index=False)