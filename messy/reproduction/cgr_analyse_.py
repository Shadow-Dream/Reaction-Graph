import pandas as pd
import pickle as pkl
import numpy as np
from cgr_analyse import analyse_dataset
from tqdm import tqdm

dataset_train = pd.read_csv("../dataset/USPTO_Condition_Train.csv")
dataset_test = pd.read_csv("../dataset/USPTO_Condition_Test.csv")
dataset_val = pd.read_csv("../dataset/USPTO_Condition_Val.csv")
#source,canonical_rxn,catalyst1,solvent1,solvent2,reagent1,reagent2,dataset
dataset = pd.concat([dataset_train,dataset_test,dataset_val])
molecules = []
for _,row in tqdm(dataset.iterrows()):
    smiles = row["canonical_rxn"]
    molecules += list(smiles.split(">>"))

result = analyse_dataset(molecules)
with open("uspto_condition_metadata.pkl","wb") as f:
    pkl.dump(result,f)

