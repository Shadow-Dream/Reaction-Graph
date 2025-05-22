import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rxnmapper import RXNMapper
from multiprocessing import Pool
import re
import os
def process(inputs):
    batch_index,dataset = inputs
    dataset["deleted"] = False
    dataset["canonical"] = ["" for _ in range(len(dataset))]
    rxn_mapper = RXNMapper()
    for index,row in dataset.iterrows():
        if (index+1) % 10000 == 0:
            with open(f"stage5.2/{batch_index}.log","w") as f:
                f.write(f"{index+1}/{len(dataset)}\n")
        
        reaction_smiles = row["smiles"]

        reaction_smiles = reaction_smiles.replace('"','')

        if '|' in reaction_smiles:
            reaction_smiles,_ = reaction_smiles.split(' ')
        pattern = re.compile(r':(\d+)]')
        if not re.findall(pattern, reaction_smiles):
            try:
                results = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])[0]
            except:
                dataset.loc[index,"deleted"] = True
                continue
            reaction_smiles = results['mapped_rxn']
        
        reaction_smiles = reaction_smiles.split('>')[0] + '>>' + reaction_smiles.split('>')[-1]

        reactants, products = reaction_smiles.split('>>')
        
        reactant_list = []
        for reactant in reactants.split('.'):
            if re.findall(pattern, reactant):
                reactant_list.append(reactant)
        reactants = '.'.join(reactant_list)
        reactant_maps = sorted(re.findall(pattern, reactants))
        product_maps = sorted(re.findall(pattern, products))
        if reactant_maps != product_maps:
            dataset.loc[index,"deleted"] = True
            continue

        mapped_reaction_smiles = reactants + '>>' + products
        dataset.loc[index,"smiles"] = mapped_reaction_smiles

        def canonicalize_smiles(smiles, clear_map=False):
            if pd.isna(smiles):
                return ''
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                if clear_map:
                    [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
                return Chem.MolToSmiles(mol)
            else:
                return ''
        
            
        reactants = canonicalize_smiles(reactants, clear_map=True)
        products = canonicalize_smiles(products, clear_map=True)
        if reactants == '' or products == '':
            dataset.loc[index,"deleted"] = True
            continue
        reaction_smiles = reactants + '>>' + products
        dataset.loc[index,"canonical"] = reaction_smiles
    dataset = dataset[~dataset["deleted"]]
    dataset = dataset.drop(columns=["deleted"])
    dataset.to_csv(f"stage5.2/{batch_index}.csv",index=False)
        

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    dataset = pd.read_csv("stage4.2/dataset.csv")
    dataset.fillna("",inplace=True)
    split_num = 16
    split_size = len(dataset) // split_num
    dataset_batches = [dataset.iloc[i*split_size:(i+1)*split_size] for i in range(split_num)]
    inputs = [(i,dataset_batches[i]) for i in range(split_num)]
    pool = Pool(split_num)
    pool.map(process,inputs)

