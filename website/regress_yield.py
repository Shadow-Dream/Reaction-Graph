from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures

from dgl.nn.pytorch import NNConv, Set2Set,SumPooling
import dgl

import pandas as pd
import numpy as np
import argparse
import warnings
import random
import json
import copy
import os

import torch
from torch import nn
from torch.nn import Module,MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from rxnmapper import RXNMapper

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = int, default = 0)
parser.add_argument("--workdir", type = str, default = "workdir")
parser.add_argument("--random_seed", type = int, default = 1234)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)
dgl.random.seed(args.random_seed)
dgl.seed(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

work_state = {
    "valid":{"progress": 0},
    "analysis": {"progress": 0},
    "preprocess": {"progress": 0},
    "train":{"progress": 0},
}

def save_state():
    with open(os.path.join(args.workdir,"state.json"),"w") as f:
        json.dump(work_state,f,indent=4)

save_state()

try:
    dataset_train = os.path.join(args.workdir,"train.csv")
    dataset_test = os.path.join(args.workdir,"test.csv")
    dataset_train = pd.read_csv(dataset_train)
    dataset_test = pd.read_csv(dataset_test)
    
    assert len(dataset_train) > 0, "Empty train dataset!"
    assert len(dataset_test) > 0, "Empty test dataset!"
    assert len(dataset_train) + len(dataset_test) <= 10000, "Too many data points!"
    assert "smiles" in dataset_train.columns, "No smiles column in train dataset!"
    assert "smiles" in dataset_test.columns, "No smiles column in test dataset!"
    assert "yield" in dataset_train.columns, "No yield column in train dataset!"
    assert "yield" in dataset_test.columns, "No yield column in test dataset!"
    
    dataset_train = dataset_train[["smiles","yield"]]
    dataset_test = dataset_test[["smiles","yield"]]
    dataset_train = dataset_train.dropna(subset = ["smiles","yield"])
    dataset_test = dataset_test.dropna(subset = ["smiles","yield"])
    dataset_train["smiles"] = dataset_train["smiles"].astype(str)
    dataset_test["smiles"] = dataset_test["smiles"].astype(str)
    dataset_train["yield"] = dataset_train["yield"].astype(float)
    dataset_test["yield"] = dataset_test["yield"].astype(float)
    
    assert len(dataset_train) > 0, "Empty train dataset!"
    assert len(dataset_test) > 0, "Empty test dataset!"
    
    dataset = pd.concat([dataset_train,dataset_test],axis=0)

except Exception as e:
    work_state["valid"]["progress"] = 1
    work_state["valid"]["result"] = False
    work_state["valid"]["error"] = str(e)
    save_state()
    exit()

work_state["valid"]["progress"] = 1
work_state["valid"]["result"] = True
save_state()

batch_size = 32

atom_set = set()
degree_set = set()
hybridization_set = set()
hydrogen_set = set()
valence_set = set()
bond_set = set()

#Analysis
valid_train = len(dataset_train)
valid_test = len(dataset_test)
total_index = 0
for index,row in dataset.iterrows():
    total_index += 1
    try:
        work_state["analysis"]["progress"] = total_index / len(dataset)
        if index % 100 == 0:
            save_state()
        smiles = row["smiles"]
        molecules = smiles.split(">")
        molecules = [Chem.MolFromSmiles(molecule) for molecule in molecules]
        for molecule in molecules:
            atom_set.update([atom.GetSymbol() for atom in molecule.GetAtoms()])
            degree_set.update([atom.GetDegree() for atom in molecule.GetAtoms()])
            hybridization_set.update([str(atom.GetHybridization()) for atom in molecule.GetAtoms()])
            hydrogen_set.update([atom.GetTotalNumHs(includeNeighbors = True) for atom in molecule.GetAtoms()])
            valence_set.update([atom.GetTotalValence() for atom in molecule.GetAtoms()])
            bond_set.update([str(bond.GetBondType()) for bond in molecule.GetBonds()])
    except:
        if index < len(dataset_train):
            valid_train -= 1
        else:
            valid_test -= 1

if valid_train == 0:
    work_state["analysis"]["result"] = False
    work_state["analysis"]["error"] = "No valid data points in train dataset!"
    save_state()
    exit()

if valid_test == 0:
    work_state["analysis"]["result"] = False
    work_state["analysis"]["error"] = "No valid data points in test dataset!"
    save_state()
    exit()

work_state["analysis"]["progress"] = 1
work_state["analysis"]["result"] = True
save_state()

ringsize_list = [3, 4, 5, 6, 7, 8]

periodic_table = Chem.GetPeriodicTable()
hybridization_table = {
"S":0,
"SP":1,
"SP2":2,
"SP3":3,
"SP2D":4,
"SP3D":5,
"SP3D2":6,
"UNSPECIFIED":7,
"OTHER":8,
}
bond_table = {	
"SINGLE":0,
"DOUBLE":1,
"TRIPLE":2,
"QUADRUPLE":3,
"QUINTUPLE":4,
"HEXTUPLE":5,
"ONEANDAHALF":6,
"TWOANDAHALF":7,
"THREEANDAHALF":8,
"FOURANDAHALF":9,
"FIVEANDAHALF":10,
"AROMATIC":11,
"IONIC":12,
"HYDROGEN":13,
"THREECENTER":14,
"DATIVEONE":15,
"DATIVE":16,
"DATIVEL":17,
"DATIVER":18,
"UNSPECIFIED":19,
"OTHER":20,
"ZERO":21
}

atom_set.update(['C','N','O','F','P','S','Cl','Br','Pd','I'])
degree_set.update([1, 2, 3, 4, 5])
hybridization_set.update(['SP2','SP3','SP3D','UNSPECIFIED'])
hydrogen_set.update([1, 2, 3, 0])
valence_set.update([1, 2, 3, 4, 5, 6, 0])
bond_set.update(['SINGLE', 'DOUBLE', 'AROMATIC'])

atom_list = sorted(list(atom_set),key = lambda x: periodic_table.GetAtomicNumber(x))
degree_list = sorted(list(degree_set))
hybridization_list = sorted(list(hybridization_set),key=lambda x: hybridization_table[x])

hydrogen_list = sorted(list(hydrogen_set))
hydrogen_list = hydrogen_list[1:] + [0]

valence_list = sorted(list(valence_set))
valence_list = valence_list[1:] + [0]

bond_list = sorted(list(bond_set),key=lambda x: bond_table[x])

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

mapper = RXNMapper()

def get_reaction_mapping(reactants,products):
    product_map_index = [0 for _ in range(len(products.GetAtoms()))]

    for reactant_atom in reactants.GetAtoms():
        map_number = reactant_atom.GetAtomMapNum()
        for product_atom in products.GetAtoms():
            if product_atom.GetAtomMapNum() == map_number:
                product_map_index[product_atom.GetIdx()] = reactant_atom.GetIdx() + 1
                break
    
    return product_map_index

def get_donor_and_acceptor_info(mol):
    donor_info, acceptor_info = [], []
    for feat in chem_feature_factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == 'Donor': donor_info.append(feat.GetAtomIds()[0])
        if feat.GetFamily() == 'Acceptor': acceptor_info.append(feat.GetAtomIds()[0])
    
    return donor_info, acceptor_info

def get_chirality_info(atom):
    return [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] if atom.HasProp('Chirality') else [0, 0]
    
def get_stereochemistry_info(bond):
    return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] if bond.HasProp('Stereochemistry') else [0, 0]    

def initialize_stereo_info(mol):
    for element in Chem.FindPotentialStereo(mol):
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': 
            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': 
            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    return mol

def canonicalize_smiles(smi):
    if pd.isna(smi):
        return ''
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''

def get_atom_3d_info(smiles,keys,slices,datas):
    molecules = smiles.split(".")
    molecules = [canonicalize_smiles(molecule) for molecule in molecules]
    molecule_keys = [keys[molecule] for molecule in molecules]
    molecule_datas = [datas[slices[key]:slices[key+1]] for key in molecule_keys]
    node_positions = np.concatenate(molecule_datas,axis=0)
    return torch.tensor(node_positions)

def get_atom_attribute(mol):
    mol = initialize_stereo_info(mol)
    donor_list, acceptor_list = get_donor_and_acceptor_info(mol) 
    atom_feature1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    # atom_feature2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_feature5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_feature6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-2]
    atom_feature7 = np.array([[(j in donor_list), (j in acceptor_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_feature8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    atom_feature9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_feature10 = np.array([get_chirality_info(a) for a in mol.GetAtoms()], dtype = bool)
    attribute = np.hstack([ atom_feature1, 
                            # atom_feature2, 
                            atom_feature3, 
                            atom_feature4, 
                            atom_feature5, 
                            atom_feature6, 
                            atom_feature7, 
                            atom_feature8, 
                            atom_feature9, 
                            atom_feature10])
    attribute = torch.tensor(attribute)
    return attribute

def get_atom_bond_attribute(mol):
    bond_feature1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
    bond_feature2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
    bond_feature3 = np.array([get_stereochemistry_info(b) for b in mol.GetBonds()], dtype = bool)
    edge_attribute = np.hstack([bond_feature1, bond_feature2, bond_feature3])
    edge_attribute = torch.tensor(edge_attribute)
    return edge_attribute

def get_reaction_graph(mapped_smiles = "",others_smiles = ""):
    reactants_smiles,products_smiles = mapped_smiles.split(">>")
    num_of_reactants_atoms = len(Chem.MolFromSmiles(reactants_smiles).GetAtoms())
    if others_smiles!="":
        reactants_smiles = f"{reactants_smiles}.{others_smiles}"

    reactant_num_atoms = [Chem.MolFromSmiles(molecule).GetNumAtoms() for molecule in reactants_smiles.split(".")]
    product_num_atoms = [Chem.MolFromSmiles(molecule).GetNumAtoms() for molecule in products_smiles.split(".")]
    reactants = Chem.MolFromSmiles(reactants_smiles)
    products = Chem.MolFromSmiles(products_smiles)

    product_map_index = get_reaction_mapping(reactants,products)
    
    num_nodes = reactants.GetNumAtoms() + products.GetNumAtoms()

    reactant_attribute = get_atom_attribute(reactants)
    product_attribute = get_atom_attribute(products)
    
    node_attribute = torch.cat([reactant_attribute,product_attribute],dim=0)
    
    node_type = torch.zeros([num_nodes,3], dtype = torch.uint8)
    node_type[:num_of_reactants_atoms,0] = 1
    node_type[num_of_reactants_atoms:reactants.GetNumAtoms(),1] = 1
    node_type[reactants.GetNumAtoms():,2] = 1

    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    message_graph.ndata['attribute'] = node_attribute.to(torch.uint8)
    message_graph.ndata['type'] = node_type.to(torch.uint8)
    
    reactant_bond_attribute = get_atom_bond_attribute(reactants)
    reactant_bond_attribute = torch.cat([reactant_bond_attribute,reactant_bond_attribute],dim=0)
    reactant_bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in reactants.GetBonds()], dtype=int)
    src = np.hstack([reactant_bonds[:,0], reactant_bonds[:,1]])
    dst = np.hstack([reactant_bonds[:,1], reactant_bonds[:,0]])
    reactant_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reactant_message_passing_graph.edata['attribute'] = reactant_bond_attribute.to(torch.uint8)

    product_bond_attribute = get_atom_bond_attribute(products)
    product_bond_attribute = torch.cat([product_bond_attribute,product_bond_attribute],dim=0)
    product_bonds = np.array([[b.GetBeginAtomIdx() + reactants.GetNumAtoms(), b.GetEndAtomIdx() + reactants.GetNumAtoms()] for b in products.GetBonds()], dtype=int)
    src = np.hstack([product_bonds[:,0], product_bonds[:,1]])
    dst = np.hstack([product_bonds[:,1], product_bonds[:,0]])
    product_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    product_message_passing_graph.edata['attribute'] = product_bond_attribute.to(torch.uint8)
    
    product_map_index = np.array(product_map_index) - 1
    product_index = np.arange(0,len(product_map_index)) + reactants.GetNumAtoms()
    edges = np.concatenate([product_map_index[:,np.newaxis],product_index[:,np.newaxis]],axis=-1)
    edges = edges[edges[:,0] != -1]
    forward_edge_attribute = np.zeros_like(edges)
    backward_edge_attribute = np.zeros_like(edges)
    forward_edge_attribute[:,0] = 1
    backward_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([forward_edge_attribute,backward_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    reaction_atom_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reaction_atom_message_passing_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    molecule_aggregation_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    num_atoms = reactant_num_atoms + product_num_atoms
    num_atoms = torch.tensor(num_atoms)
    molecule_aggregation_graph.set_batch_num_nodes(num_atoms)
    molecule_aggregation_graph.set_batch_num_edges(torch.zeros_like(num_atoms))

    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    reaction_aggregation_graph = dgl.graph((src, dst), num_nodes = len(num_atoms))
    num_nodes = torch.tensor([len(reactant_num_atoms),len(product_num_atoms)])
    reaction_aggregation_graph.set_batch_num_nodes(num_nodes)
    reaction_aggregation_graph.set_batch_num_edges(torch.zeros_like(num_nodes))
    
    return {"message_graph":message_graph,
            "reactant_message_passing_graph":reactant_message_passing_graph,
            "product_message_passing_graph":product_message_passing_graph,
            "reaction_atom_message_passing_graph":reaction_atom_message_passing_graph,
            "molecule_aggregation_graph":molecule_aggregation_graph,
            "reaction_aggregation_graph":reaction_aggregation_graph}

total_length = len(dataset)
total_index = 0
def preprocess(dataset):
    global total_index
    batches = []
    key_list = []
    input_list = []
    output_list = []
            
    for index,row in dataset.iterrows():
        total_index += 1
        work_state["preprocess"]["progress"] = total_index / total_length
        if total_index % 100 == 0:
            save_state()
        try:
            smiles = row["smiles"]
            reactant,reagent,product = smiles.split(">")
            others_smiles = reagent
            mapped_smiles = mapper.get_attention_guided_atom_maps([f"{reactant}>>{product}"])[0]['mapped_rxn']
            reaction_yield = row["yield"]
        
            return_dict = get_reaction_graph(mapped_smiles,others_smiles)
            if return_dict is None:
                raise Exception("Empty return dict!")
            key_list.append(smiles)
            input_list.append(return_dict)
            output_list.append(reaction_yield)
        except Exception:
            continue
        if index % batch_size == batch_size - 1:
            input_dict = {key:[item[key] for item in input_list] for key in input_list[0]}
            input_dict["message_graph"] = dgl.batch(input_dict["message_graph"])
            input_dict["reactant_message_passing_graph"] = dgl.batch(input_dict["reactant_message_passing_graph"])
            input_dict["product_message_passing_graph"] = dgl.batch(input_dict["product_message_passing_graph"])
            input_dict["reaction_atom_message_passing_graph"] = dgl.batch(input_dict["reaction_atom_message_passing_graph"])
            input_dict["molecule_aggregation_graph"] = dgl.batch(input_dict["molecule_aggregation_graph"])
            input_dict["reaction_aggregation_graph"] = dgl.batch(input_dict["reaction_aggregation_graph"])

            batch = {}
            batch["keys"] = copy.deepcopy(key_list)
            batch["inputs"] = input_dict
            batch["outputs"] = torch.tensor(output_list)
            batches.append(batch)
            key_list = []
            input_list = []
            output_list = []
    
    if len(key_list) > 0:
        input_dict = {key:[item[key] for item in input_list] for key in input_list[0]}
        input_dict["message_graph"] = dgl.batch(input_dict["message_graph"])
        input_dict["reactant_message_passing_graph"] = dgl.batch(input_dict["reactant_message_passing_graph"])
        input_dict["product_message_passing_graph"] = dgl.batch(input_dict["product_message_passing_graph"])
        input_dict["reaction_atom_message_passing_graph"] = dgl.batch(input_dict["reaction_atom_message_passing_graph"])
        input_dict["molecule_aggregation_graph"] = dgl.batch(input_dict["molecule_aggregation_graph"])
        input_dict["reaction_aggregation_graph"] = dgl.batch(input_dict["reaction_aggregation_graph"])
        batch = {}
        batch["keys"] = copy.deepcopy(key_list)
        batch["inputs"] = input_dict
        batch["outputs"] = torch.tensor(output_list)
        batches.append(batch)

    return batches

train_batches = preprocess(dataset_train)
test_batches = preprocess(dataset_test)

work_state["preprocess"]["progress"] = 1
work_state["preprocess"]["result"] = True
save_state()

class Dataset:
    def __init__(self, 
                 batches, 
                 device = "cuda",
                 shuffle = True):
        self.dataset_index = 0
        self.dataset_offset = -1
        self.dataset_buffer = batches
        self.device = device
        self.postprocess_dataset()
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_dataset()
        
        
    def __len__(self):
        return len(self.dataset_buffer)

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def postprocess_dataset(self):
        output_list = []
        for batch in self.dataset_buffer:
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            message_graph = inputs["message_graph"]
            reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
            product_message_passing_graph = inputs["product_message_passing_graph"]
            reaction_atom_message_passing_graph = inputs["reaction_atom_message_passing_graph"]
            molecule_aggregation_graph = inputs["molecule_aggregation_graph"]
            reaction_aggregation_graph = inputs["reaction_aggregation_graph"]

            num_atoms_each_batch = message_graph.batch_num_nodes()

            num_edges_each_batch = 0
            num_edges_each_batch += reactant_message_passing_graph.batch_num_edges()
            num_edges_each_batch += product_message_passing_graph.batch_num_edges()
            num_edges_each_batch += reaction_atom_message_passing_graph.batch_num_edges()

            reactant_atom_edge_attribute = reactant_message_passing_graph.edata["attribute"]
            product_atom_edge_attribute = product_message_passing_graph.edata["attribute"]
            reaction_atom_edge_attribute = reaction_atom_message_passing_graph.edata["attribute"]

            reactant_atom_num_edge = reactant_message_passing_graph.num_edges()
            product_atom_num_edge = product_message_passing_graph.num_edges()
            reaction_atom_num_edge = reaction_atom_message_passing_graph.num_edges()

            dim_edge_attribute = reactant_atom_edge_attribute.size(1)

            reactant_atom_padding = torch.zeros([reactant_atom_num_edge,2])
            reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

            product_atom_padding = torch.zeros([product_atom_num_edge,2])
            product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

            reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute])
            reaction_atom_edge_attribute = torch.cat([reaction_atom_padding_left,reaction_atom_edge_attribute],dim = -1)

            reactant_message_passing_graph.edata["attribute"] = reactant_atom_edge_attribute
            product_message_passing_graph.edata["attribute"] = product_atom_edge_attribute
            reaction_atom_message_passing_graph.edata["attribute"] = reaction_atom_edge_attribute

            message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reaction_atom_message_passing_graph])
            message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_edges(num_edges_each_batch)

            inputs = {}
            inputs["message_graph"] = message_graph
            inputs["message_passing_graph"] = message_passing_graph
            inputs["molecule_aggregation_graph"] = molecule_aggregation_graph
            inputs["reaction_aggregation_graph"] = reaction_aggregation_graph
            batch["inputs"] = inputs
            batch["outputs"] = outputs
            output_list.append(outputs)
        outputs = np.concatenate(output_list)
        self.mean = np.mean(outputs)
        self.std = np.std(outputs)
        self.var = np.var(outputs)
        self.outputs = outputs
        self.dim_node_attribute = batch["inputs"]["message_graph"].ndata["attribute"].size(-1)
        self.dim_edge_attribute = batch["inputs"]["message_passing_graph"].edata["attribute"].size(-1)

    def process(self,batch):
        inputs = batch["inputs"]
        outputs = torch.tensor(batch["outputs"]).float().to(self.device)
        
        message_graph = inputs["message_graph"].to(self.device)
        message_passing_graph = inputs["message_passing_graph"].to(self.device)
        molecule_aggregation_graph = inputs["molecule_aggregation_graph"].to(self.device)
        reaction_aggregation_graph = inputs["reaction_aggregation_graph"].to(self.device)

        message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
        message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
        
        inputs = {"message_graph":message_graph,
                  "message_passing_graph":message_passing_graph,
                  "molecule_aggregation_graph":molecule_aggregation_graph,
                  "reaction_aggregation_graph":reaction_aggregation_graph}
        
        new_batch = {"inputs":inputs,"outputs":outputs}
        return new_batch

    def __iter__(self):
        return self
            
    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = -1
            if self.shuffle:
                self.shuffle_dataset()
            raise StopIteration
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch

class MPNN(Module):
    def __init__(self,
                 dim_node_attribute,
                 dim_edge_attribute,
                 dim_hidden_features,
                 dim_hidden,
                 message_passing_step,
                 pooling_step,
                 num_layers_pooling):
        
        super(MPNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(
            nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
    
        self.bond_function = nn.Linear(dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
    
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)

        self.pooling1 = Set2Set(input_dim = dim_hidden_features * 2, 
                               n_iters = pooling_step, 
                               n_layers = num_layers_pooling)
        self.pooling2 = SumPooling()
        self.sparsify = nn.Sequential(
            nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU())

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]
        molecule_aggregation_graph = inputs["molecule_aggregation_graph"]
        reaction_aggregation_graph = inputs["reaction_aggregation_graph"]

        node_attribute = message_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        
        edge_features = message_passing_graph.edata["attribute"]
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(message_passing_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)

        reaction_features = self.pooling1(molecule_aggregation_graph, node_aggregation)
        reaction_features = self.sparsify(reaction_features)
        reaction_features = self.pooling2(reaction_aggregation_graph, reaction_features)
        reaction_features = reaction_features.reshape(message_passing_graph.batch_size, -1)
        return reaction_features
    
class Network(Module):
    def __init__(self,
                 dim_node_attribute = 43,
                 dim_edge_attribute = 10,
                 dim_hidden_features = 64,
                 dim_hidden = 1024,
                 message_passing_step = 3,
                 pooling_step = 3,
                 num_layers_pooling = 1,
                 dim_hidden_regression = 512,
                 dropout = 0.1):
        
        super(Network, self).__init__()
        
        self.mpnn = MPNN(dim_node_attribute = dim_node_attribute,
                         dim_edge_attribute = dim_edge_attribute,
                         dim_hidden_features = dim_hidden_features,
                         dim_hidden = dim_hidden,
                         message_passing_step = message_passing_step,
                         pooling_step = pooling_step,
                         num_layers_pooling = num_layers_pooling)
        self.regression = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, 2)
        )
        
    def forward(self, inputs):
        reaction_features = self.mpnn(inputs)
        outputs = self.regression(reaction_features)
        mean = outputs[:,0]
        logvar = outputs[:,1]
        return mean, logvar

def training(network, 
             dataset_train, 
             dataset_test, 
             train_epoches = 500,
             num_inference_pass = 5,
             workdir = ""):
    criterion = MSELoss(reduction = 'none')

    optimizer = Adam(network.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [400, 450], gamma = 0.1, verbose = False)
    max_r2 = 0
    for epoch in range(train_epoches):
        network.train()
        losses = []
        for index,batch in enumerate(dataset_train):
            inputs = batch["inputs"]
            real_mean = batch["outputs"]
            real_mean = (real_mean - dataset_train.mean) / dataset_train.std
            
            pred_mean,pred_var = network(inputs)

            loss = criterion(pred_mean, real_mean)
            loss = 0.9 * loss.mean() + 0.1 * (loss * torch.exp(-pred_var) + pred_var).mean()
            loss /= 4
            loss.backward()
            if index % 4==3:
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.detach().item()*4)
        losses = np.mean(losses).item()
        learning_rate = optimizer.param_groups[0]['lr']

        lr_scheduler.step()
        pred_mean,_,_ = inference(network,dataset_test,dataset_train.mean,dataset_train.std,num_inference_pass)
        real_mean = dataset_test.outputs
        mae = mean_absolute_error(real_mean, pred_mean)
        rmse = mean_squared_error(real_mean, pred_mean) ** 0.5
        r2 = r2_score(real_mean, pred_mean)
        work_state["train"]["progress"] = epoch / train_epoches
        work_state["train"]["loss"] = losses
        work_state["train"]["r2"] = max_r2
        save_state()
        if r2 > max_r2:
            max_r2 = r2
            state_dict = {
                "dim_node_attribute": dataset_train.dim_node_attribute,
                "dim_edge_attribute": dataset_train.dim_edge_attribute,
                "atom_list": atom_list,
                "degree_list": degree_list,
                "hybridization_list": hybridization_list,
                "hydrogen_list": hydrogen_list,
                "valence_list": valence_list,
                "bond_list": bond_list,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mean": dataset_train.mean,
                "std": dataset_train.std,
                "model":network.state_dict()
            }
            torch.save(state_dict, os.path.join(workdir,f"model.pth"))
    work_state["train"]["progress"] = 1
    work_state["train"]["result"] = True
    save_state()
    return network
    
@torch.no_grad()
def inference(network, dataset_test, mean, std, num_inference_pass = 30):        
    network.eval()
    for m in network.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    test_mean = []
    test_var = []
    
    for batch in dataset_test:
        mean_list = []
        var_list = []
        inputs = batch["inputs"]
        
        for _ in range(num_inference_pass):
            pred_mean, pred_logvar = network(inputs)
            mean_list.append(pred_mean.cpu().numpy())
            var_list.append(np.exp(pred_logvar.cpu().numpy()))

        test_mean.append(np.array(mean_list).transpose())
        test_var.append(np.array(var_list).transpose())

    test_mean = np.vstack(test_mean) * std + mean
    test_var = np.vstack(test_var) * std ** 2
    
    prediction = np.mean(test_mean, 1)
    epistemic = np.var(test_mean, 1)
    aleatoric = np.mean(test_var, 1)
    
    return prediction, epistemic, aleatoric

dataset_train = Dataset(train_batches, shuffle=True)
dataset_test = Dataset(test_batches, shuffle=False)

network = Network(
    dim_node_attribute=dataset_train.dim_node_attribute,
    dim_edge_attribute=dataset_train.dim_edge_attribute,
    dim_hidden_features = 64,
    dim_hidden = 1024,
    message_passing_step = 3,
    pooling_step = 3,
    num_layers_pooling = 1,
    dim_hidden_regression = 512,
    dropout = 0.1
).cuda()

network = training(network, 
                   dataset_train, 
                   dataset_test, 
                   train_epoches = 500, 
                   num_inference_pass = 5,
                   workdir = args.workdir)
