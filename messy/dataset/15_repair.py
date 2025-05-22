import pandas as pd
import pickle as pkl
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file",type = str,default="graph/PistachioConditionTest0_9600.pkl")
args = parser.parse_args()
offset = 0
file = args.file
with open(file,"rb") as f:
    batches = pkl.load(f)
if "Train" in file:
    dataset = pd.read_csv('final/train.csv')
elif "Test" in file:
    dataset = pd.read_csv('final/test.csv')
elif "Val" in file:
    dataset = pd.read_csv('final/val.csv')
mapped_smiles_list = []
for batch in tqdm(batches):
    for key in batch["keys"]:
        while dataset.loc[offset,"canonical_smiles"] != key:
            offset += 1
        mapped_smiles = dataset.loc[offset,"mapped_smiles"]
        mapped_smiles_list.append(mapped_smiles)
        offset += 1
with open(file.replace("graph","mapped_smiles").replace(".pkl",".txt"),"w") as f:
    f.write("\n".join(mapped_smiles_list))