from rxnmapper import RXNMapper
import re
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_num", type=int, default=64)
parser.add_argument("--split_index", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
split_num = args.split_num
split_index = args.split_index

def canonicalize_smiles(smi, clear_map=True):
    if pd.isna(smi):
        return ''
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if clear_map:
            [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''

reaction_type_dataset = {"mapped_smiles": [], "origin_smiles": [],"others_smiles":[], "category": []}
rxnmapper = RXNMapper()
pattern = re.compile(r':(\d+)]')
with open("pistachio.cansmi","r") as f:
    lines = f.readlines()
split_size = len(lines) // split_num + 1
lines = lines[split_index*split_size:(split_index+1)*split_size]
errors = []
for line in tqdm(lines):
    try:
        parts = line.split("\t")
        smiles = parts[0]
        category = parts[3]

        origin_smiles = smiles

        if "|" in smiles:
            smiles = smiles.split(" ")[0]
        parts = smiles.split(">")
        reactants = parts[0]
        others = parts[1:-1]
        products = parts[-1]
        
        smiles = reactants + ">>" + products
        smiles = rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
        reactants, products = smiles.split(">>")
        reactant_list = []
        others = sum([other.split(".") for other in others], [])
        others = [other for other in others if other!=""]

        for reactant in reactants.split("."):
            if re.findall(pattern, reactant):
                reactant_list.append(reactant)
            else:
                others.append(reactant)
        reactants = ".".join(reactant_list)
        reactant_maps = sorted(re.findall(pattern, reactants))
        product_maps = sorted(re.findall(pattern, products))
        if reactant_maps != product_maps:
            raise ValueError("Maps do not match")
        smiles = reactants + ">>" + products
        mapped_smiles = smiles
        others_smiles = ".".join(others)
        reaction_type_dataset["mapped_smiles"].append(mapped_smiles)
        reaction_type_dataset["origin_smiles"].append(origin_smiles)
        reaction_type_dataset["others_smiles"].append(others_smiles)
        reaction_type_dataset["category"].append(category)
    except:
        errors.append(line)
        continue

reaction_type_dataset = pd.DataFrame(reaction_type_dataset)
reaction_type_dataset.to_csv(f"reaction_type_dataset/reaction_type_dataset_{split_index}.csv", index=False)
with open(f"reaction_type_dataset/errors_{split_index}.txt", "w") as f:
    f.write("\n".join(errors))