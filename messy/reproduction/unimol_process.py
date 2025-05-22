import pickle as pkl
import dgl
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from unimol_tools.models import UniMolModel
from unimol_tools import UniMolRepr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start",type=int)
parser.add_argument("--start_1",type=int)
parser.add_argument("--end",type=int)
parser.add_argument("--end_1",type=int)
args = parser.parse_args()
# 70400
model = UniMolModel(output_dim=1, data_type="molecule", remove_hs=True)
batch_size = 16
dataset = pd.read_csv("/xxx/chem/USPTO_Condition/USPTO_Condition_Val.csv")
solvents = f"/xxx/chem/USPTO_Condition/USPTO_Condition_Solvent.pkl"
catalysts = f"/xxx/chem/USPTO_Condition/USPTO_Condition_Catalyst.pkl"
reagents = f"/xxx/chem/USPTO_Condition/USPTO_Condition_Reagent.pkl"
with open(solvents,"rb") as f:
    solvents = np.array(pkl.load(f))
with open(catalysts,"rb") as f:
    catalysts = np.array(pkl.load(f))
with open(reagents,"rb") as f:
    reagents = np.array(pkl.load(f))
def encode_condition(row):
    catalyst1 = str(row["catalyst1"]) == catalysts
    solvent1 = str(row["solvent1"]) == solvents
    solvent2 = str(row["solvent2"]) == solvents
    reagent1 = str(row["reagent1"]) == reagents
    reagent2 = str(row["reagent2"]) == reagents
    encoding = np.concatenate([catalyst1,solvent1,solvent2,reagent1,reagent2]).astype(np.byte)
    return encoding

with open("/xxx/chem/USPTO_Condition/Unimol_Val_Conformer.pkl","rb") as f:
    unimol_input = pkl.load(f)
batches = []
batch_conformers = []
batch_graphs = []
batch_outputs = []
dataset = dataset[args.start:args.end]
unimol_input = unimol_input[args.start_1:args.end_1]
for index,row in tqdm(dataset.iterrows(),total=len(dataset)):
    reaction = row["canonical_rxn"]
    reactants,products = reaction.split('>>')
    reactants = reactants.split('.')
    products = products.split('.')
    molecules = reactants + products
    output = encode_condition(row)
    conformers = unimol_input[:len(molecules)]
    unimol_input = unimol_input[len(molecules):]
    conformers = [(conformer,) for conformer in conformers]
    is_next = False
    for conformer in conformers:
        if (conformer[0]["src_coord"]==0.0).all() and len(conformer[0]["src_tokens"])!=3:
            is_next = True
            break
    if is_next:
        continue
    batch_conformers += conformers
    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    reactant_count = sum([len(conformers[i][0]["src_tokens"])-2 for i in range(len(reactants))])
    product_count = sum([len(conformers[i+len(reactants)][0]["src_tokens"])-2 for i in range(len(products))])
    reactant_graph = dgl.graph((src,dst),num_nodes = reactant_count)
    product_graph = dgl.graph((src,dst),num_nodes = product_count)
    batch_graphs += [reactant_graph,product_graph]
    batch_outputs.append(output)
    if (len(batch_outputs) + 1) % batch_size == 0:
        batch = model.batch_collate_fn(batch_conformers)[0]
        batch["graphs"] = dgl.batch(batch_graphs)
        batch["outputs"] = np.array(batch_outputs)
        batches.append(batch)
        batch_graphs = []
        batch_conformers = []
        batch_outputs = []
if len(batch_outputs)>0:
    batch = model.batch_collate_fn(batch_conformers)[0]
    batch["graphs"] = dgl.batch(batch_graphs)
    batch["outputs"] = np.array(batch_outputs)
    batches.append(batch)
with open(f"/xxx/chem/USPTO_Condition/Unimol_Val_{args.start}_{args.end}.pkl","wb") as f:
    pkl.dump(batches,f)
# Test 35200 101687
# Val 35200 101401
#nohup python process.py --start 0      --end 35200  --start_1 0       --end_1 101401  > logs/00.log 2>&1 &
#nohup python process.py --start 35200  --end 70400  --start_1 101401  --end_1 900000  > logs/01.log 2>&1 &
#nohup python process.py --start 70400  --end 105600 --start_1 202759  --end_1 304182  > logs/02.log 2>&1 &
#nohup python process.py --start 105600 --end 140800 --start_1 304182  --end_1 405674  > logs/03.log 2>&1 &
#nohup python process.py --start 140800 --end 176000 --start_1 405674  --end_1 506993  > logs/04.log 2>&1 &
#nohup python process.py --start 176000 --end 211200 --start_1 506993  --end_1 608526  > logs/05.log 2>&1 &
#nohup python process.py --start 211200 --end 246400 --start_1 608526  --end_1 710022  > logs/06.log 2>&1 &
#nohup python process.py --start 246400 --end 281600 --start_1 710022  --end_1 811528  > logs/07.log 2>&1 &
#nohup python process.py --start 281600 --end 316800 --start_1 811528  --end_1 913017  > logs/08.log 2>&1 &
#nohup python process.py --start 316800 --end 352000 --start_1 913017  --end_1 1014700 > logs/09.log 2>&1 &
#nohup python process.py --start 352000 --end 387200 --start_1 1014700 --end_1 1116146 > logs/10.log 2>&1 &
#nohup python process.py --start 387200 --end 422400 --start_1 1116146 --end_1 1217629 > logs/11.log 2>&1 &
#nohup python process.py --start 422400 --end 457600 --start_1 1217629 --end_1 1319277 > logs/12.log 2>&1 &
#nohup python process.py --start 457600 --end 492800 --start_1 1319277 --end_1 1420886 > logs/13.log 2>&1 &
#nohup python process.py --start 492800 --end 528000 --start_1 1420886 --end_1 1522099 > logs/14.log 2>&1 &
#nohup python process.py --start 528000 --end 563200 --start_1 1522099 --end_1 3000000 > logs/15.log 2>&1 &