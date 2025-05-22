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
parser.add_argument("--reactant", type = str, default = "Brc1cccnc1.Cc1ccc(N)cc1")
parser.add_argument("--reagent", type = str, default = "CC(C)c1cc(C(C)C)c(-c2ccccc2[PH](C(C)(C)C)(C(C)(C)C)[Pd]2(OS(=O)(=O)C(F)(F)F)Nc3ccccc3-c3ccccc32)c(C(C)C)c1.CN(C)C(=NC(C)(C)C)N(C)C.COC(=O)c1cc(-c2cccs2)on1.CS(C)=O")
parser.add_argument("--product", type = str, default = "Cc1ccc(Nc2cccnc2)cc1")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

state_dict = os.path.join(args.workdir,"model.pth")
state_dict = torch.load(state_dict)
atom_list = state_dict["atom_list"]
degree_list = state_dict["degree_list"]
hybridization_list = state_dict["hybridization_list"]
hydrogen_list = state_dict["hydrogen_list"]
valence_list = state_dict["valence_list"]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = state_dict["bond_list"]
mean = state_dict["mean"]
std = state_dict["std"]
model = state_dict["model"]

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

try:
    reactant,reagent,product = args.reactant,args.reagent,args.product
    others_smiles = reagent
    mapped_smiles = mapper.get_attention_guided_atom_maps([f"{reactant}>>{product}"])[0]['mapped_rxn']
    workdir = args.workdir
    with open(f"{workdir}/mapped_smiles.json","w") as f:
        json.dump({"smiles":mapped_smiles},f)
    inputs = get_reaction_graph(mapped_smiles,others_smiles)
    inputs = {key:[value] for key,value in inputs.items()}
    inputs["message_graph"] = dgl.batch(inputs["message_graph"])
    inputs["reactant_message_passing_graph"] = dgl.batch(inputs["reactant_message_passing_graph"])
    inputs["product_message_passing_graph"] = dgl.batch(inputs["product_message_passing_graph"])
    inputs["reaction_atom_message_passing_graph"] = dgl.batch(inputs["reaction_atom_message_passing_graph"])
    inputs["molecule_aggregation_graph"] = dgl.batch(inputs["molecule_aggregation_graph"])
    inputs["reaction_aggregation_graph"] = dgl.batch(inputs["reaction_aggregation_graph"])

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

    message_graph = message_graph.to("cuda")
    message_passing_graph = message_passing_graph.to("cuda")
    molecule_aggregation_graph = molecule_aggregation_graph.to("cuda")
    reaction_aggregation_graph = reaction_aggregation_graph.to("cuda")

    message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
    message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
    
    inputs = {"message_graph":message_graph,
                "message_passing_graph":message_passing_graph,
                "molecule_aggregation_graph":molecule_aggregation_graph,
                "reaction_aggregation_graph":reaction_aggregation_graph}
except:
    exit()

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

network = Network(
    dim_node_attribute=state_dict["dim_node_attribute"],
    dim_edge_attribute=state_dict["dim_edge_attribute"],
    dim_hidden_features = 64,
    dim_hidden = 1024,
    message_passing_step = 3,
    pooling_step = 3,
    num_layers_pooling = 1,
    dim_hidden_regression = 512,
    dropout = 0.1
).cuda()

network.load_state_dict(model)
network.eval()

for m in network.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()

mean_list = []
var_list = []

with torch.no_grad():
    for _ in range(30):
        pred_mean, pred_logvar = network(inputs)
        mean_list.append(pred_mean.cpu().numpy())
        var_list.append(np.exp(pred_logvar.cpu().numpy()))

mean_list = np.array(mean_list) * std + mean
var_list = np.array(var_list) * (std ** 2)

var = var_list.mean() + mean_list.var()
mean = mean_list.mean()

result = {"yield":float(mean),"variance":float(var)}
workdir = args.workdir
with open(f"{workdir}/result.json","w") as f:
    json.dump(result,f)