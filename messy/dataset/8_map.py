import pandas as pd
import json
from tqdm import tqdm
import argparse
from rxnmapper import RXNMapper
import os
import re
from rdkit import Chem

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return ''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''

parser = argparse.ArgumentParser()
parser.add_argument("--split_index",type=int,default=0)
args = parser.parse_args()
split_index = args.split_index
split_num = 20
device = split_index % 2 + 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

with open("agents.txt") as f:
    agents = f.read().split("\n")

dataset = pd.read_csv("condition/stage7/dataset.csv")
split_batch = len(dataset) // split_num + 1
dataset = dataset.iloc[split_batch*split_index:split_batch*(split_index+1)]
dataset.fillna("",inplace=True)
dataset.reset_index(drop=True,inplace=True)
pattern = re.compile(r':(\d+)]')
rxnmapper = RXNMapper()
dataset["mapped_smiles"] = ""
dataset["canonical_smiles"] = ""
dataset["invalid"] = False
for index,row in tqdm(dataset.iterrows(),total=len(dataset)):
    smiles = row["smiles"]
    try:
        mapped_smiles = rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
        reactants,products = mapped_smiles.split(">>")
        reactants = reactants.split(".")
        new_agents = []
        new_reactants = []
        for reactant in reactants:
            if re.findall(pattern, reactant):
                new_reactants.append(reactant)
            else:
                new_agents.append(reactant)

        if len(new_agents) == 0:
            pass
        elif len(new_agents) == 1:
            if row["agent1"] == "":
                dataset.loc[index,"agent1"] = new_agents[0]
            elif row["agent2"] == "":
                dataset.loc[index,"agent2"] = new_agents[0]
            else:
                dataset.loc[index,"invalid"] = True
                continue
        elif len(new_agents) == 2:
            if row["agent1"] == "":
                dataset.loc[index,"agent1"] = new_agents[0]
                dataset.loc[index,"agent2"] = new_agents[1]
            else:
                dataset.loc[index,"invalid"] = True
                continue
        else:
            dataset.loc[index,"invalid"] = True
            continue

        invalid = False
        for new_agent in new_agents:
            if new_agent not in agents:
                invalid = True
                break

        if invalid:
            dataset.loc[index,"invalid"] = True
            continue

        reactants = new_reactants
        reactants = ".".join(reactants)
        reactant_maps = sorted(re.findall(pattern, reactants))
        product_maps = sorted(re.findall(pattern, products))
        if reactant_maps != product_maps:
            dataset.loc[index,"invalid"] = True
            continue
        
        reaction = reactants + ">>" + products
        dataset.loc[index,"mapped_smiles"] = reaction
        reactants = canonicalize_smiles(reactants)
        products = canonicalize_smiles(products)
        if reactants == "" or products == "":
            dataset.loc[index,"invalid"] = True
            continue
        reaction = reactants + ">>" + products
        dataset.loc[index,"canonical_smiles"] = reaction
    except:
        dataset.loc[index,"invalid"] = True

dataset = dataset[~dataset["invalid"]]
#刪除invalid列
dataset.drop(columns=["invalid"],inplace=True)
dataset.to_csv(f"condition/stage8/dataset_{split_index}.csv",index=False)

