import os
import warnings
import argparse
import numpy as np

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures,AllChem,rdChemReactions

import dgl
from dgl.ops import segment
from dgl.nn.pytorch import NNConv, SumPooling

import torch
from torch import nn
from torch.nn import Module

import json

from rxnmapper import RXNMapper
from itertools import groupby
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
atom_list = ["C","O","S","N","F",
             "Cl","Br","K","B","Pd",
             "P","H","Si","Na","Cs",
             "I","Cu","Li","Mg","Fe",
             "Ni","Al","Sn","Pt","Ti",
             "Co","Zn","Rh","Ce","Ca",
             "Hg","Ag","Yb","Ru","Mn",
             "Dy","Ba","Cr","Se","Bi",
             "Os","Sc","In","Mo","Sm",
             "Pb","As","Ir","Zr","La",
             "Pr","V","Au","Xe","W",
             "Ge","Tl","Y","Ga","Hf",
             "Rb","*","Gd","Re","Ar",
             "Nd","Te","Sb"]
# 68

#元素所含的电荷
charge_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]# 11
#原子键度数
degree_list = [1, 2, 3, 4, 5, 6, 7, 0]# 8
#杂化类型
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']# 7
#氢原子数量
hydrogen_list = [1, 2, 3, 4, 0]# 5
#化合价
valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]# 13
#环大小
ringsize_list = [3, 4, 5, 6, 7, 8]
#可能的化学键
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'UNSPECIFIED']

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

        self.pooling = SumPooling()
        self.sparsify = nn.Sequential(
            nn.Linear(dim_hidden_features * 2, dim_hidden), nn.PReLU())

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]
        molecule_aggregation_graph = inputs["molecule_aggregation_graph"]
        num_reactant_batch = inputs["num_reactant_batch"]
        num_product_batch = inputs["num_product_batch"]
        select_reactant = inputs["select_reactant"]

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

        reaction_features = self.pooling(molecule_aggregation_graph, node_aggregation)
        reaction_features = self.sparsify(reaction_features)
        reactant_features = segment.segment_reduce(num_reactant_batch, reaction_features[select_reactant], reducer="sum")
        product_features = segment.segment_reduce(num_product_batch, reaction_features[~select_reactant], reducer="sum")
        reaction_features = torch.cat([reactant_features,product_features],dim = -1)
        return reaction_features
    
class Network(Module):
    def __init__(self,
                dim_node_attribute = 121,
                dim_edge_attribute = 14,
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

def get_reaction_center_and_mapping(reaction,reactants,products):
    reactant_center_atoms = reaction.GetReactingAtoms()

    reactant_offsets = [0]
    for reactant_molecule in reaction.GetReactants()[:-1]:
        reactant_offsets.append(reactant_offsets[-1] + reactant_molecule.GetNumAtoms())

    reactant_center_atoms = [reactant_atom + reactant_offset
                            for reactant_offset,reactant_atoms 
                            in zip(reactant_offsets,reactant_center_atoms)
                            for reactant_atom 
                            in reactant_atoms]
    
    product_center_atoms = []
    for reactant_center_atom in reactant_center_atoms:
        map_number = reactants.GetAtomWithIdx(reactant_center_atom).GetAtomMapNum()
        for product_atom_index,product_atom in enumerate(products.GetAtoms()):
            if product_atom.GetAtomMapNum() == map_number:
                product_center_atoms.append(product_atom_index)
                break
    
    for atom_index,atom in enumerate(reactants.GetAtoms()):
        if atom.GetAtomMapNum() == 0:
            reactant_center_atoms.append(atom_index)

    for atom_index,atom in enumerate(products.GetAtoms()):
        if atom.GetAtomMapNum() == 0:
            product_center_atoms.append(atom_index)
    
    reactant_map_index = [0 for _ in range(len(reactants.GetAtoms()))]
    product_map_index = [0 for _ in range(len(products.GetAtoms()))]

    for reactant_atom in reactants.GetAtoms():
        map_number = reactant_atom.GetAtomMapNum()
        for product_atom in products.GetAtoms():
            if product_atom.GetAtomMapNum() == map_number:
                reactant_map_index[reactant_atom.GetIdx()] = product_atom.GetIdx() + 1
                product_map_index[product_atom.GetIdx()] = reactant_atom.GetIdx() + 1
                break
    
    reactant_center_atoms = list(set(reactant_center_atoms))
    product_center_atoms = list(set(product_center_atoms))
    return reactant_center_atoms,product_center_atoms,reactant_map_index,product_map_index

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

def get_atom_attribute(mol):
    mol = initialize_stereo_info(mol)
    donor_list, acceptor_list = get_donor_and_acceptor_info(mol) 
    atom_feature1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_feature2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_feature5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_feature6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-2]
    atom_feature7 = np.array([[(j in donor_list), (j in acceptor_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_feature8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    atom_feature9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_feature10 = np.array([get_chirality_info(a) for a in mol.GetAtoms()], dtype = bool)
    attribute = np.hstack([ atom_feature1, 
                            atom_feature2, 
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

def get_atom_fingerprint(mol,fingerprint_radius = 2):
    node_fingerprint = [[] for _ in range(mol.GetNumAtoms())]
    info = {}
    AllChem.GetMorganFingerprint(mol, fingerprint_radius, bitInfo=info)
    for fingerprint_index,value in info.items():
        for atom_index,_ in value:
            node_fingerprint[atom_index].append(fingerprint_index)
    return node_fingerprint

def get_molecule_fingerprint(mols,fingerprint_radius = 2):
    attributes = []
    for mol in mols:
        info = {}
        mol.UpdatePropertyCache()
        Chem.GetSSSR(mol)
        AllChem.GetMorganFingerprint(mol, fingerprint_radius, bitInfo=info)
        attributes.append(list(info.keys()))
    return attributes

def get_atom_bond_attribute(mol):
    bond_feature1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
    bond_feature2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
    bond_feature3 = np.array([get_stereochemistry_info(b) for b in mol.GetBonds()], dtype = bool)
    edge_attribute = np.hstack([bond_feature1, bond_feature2, bond_feature3])
    edge_attribute = torch.tensor(edge_attribute)
    return edge_attribute

def get_reaction_graph(smiles = ""):
    reactants_smiles,products_smiles = smiles.split(">>")
    reactants = Chem.MolFromSmiles(reactants_smiles)
    products = Chem.MolFromSmiles(products_smiles)

    if reactants is None or products is None:
        raise Exception("Invalid SMILES string")

    reaction = rdChemReactions.ReactionFromSmarts(smiles)
    reaction.Initialize()

    (reactant_center_atoms,
     product_center_atoms,
     reactant_map_index,
     product_map_index) = get_reaction_center_and_mapping(reaction,reactants,products)
    
    num_atom_nodes = reactants.GetNumAtoms() + products.GetNumAtoms()
    num_molecule_nodes = len(reaction.GetReactants()) + len(reaction.GetProducts())
    num_nodes = num_atom_nodes + num_molecule_nodes

    reactant_attribute = get_atom_attribute(reactants)
    product_attribute = get_atom_attribute(products)

    reactant_fingerprint = get_atom_fingerprint(reactants)
    product_fingerprint = get_atom_fingerprint(products)
    
    node_attribute_padding = torch.stack([torch.zeros_like(reactant_attribute[0]) ]* num_molecule_nodes)

    reactant_molecule_fingerprint = get_molecule_fingerprint(reaction.GetReactants())
    product_molecule_fingerprint = get_molecule_fingerprint(reaction.GetProducts())

    node_attribute = torch.cat([reactant_attribute,product_attribute,node_attribute_padding],dim=0)

    node_type = torch.zeros([num_nodes,4], dtype = torch.uint8)
    node_type[:reactants.GetNumAtoms(),0] = 1
    node_type[reactants.GetNumAtoms():num_atom_nodes,1] = 1
    node_type[num_atom_nodes:num_atom_nodes + len(reaction.GetReactants()),2] = 1
    node_type[num_atom_nodes + len(reaction.GetReactants()):,3] = 1

    reactant_center_atoms = np.array(reactant_center_atoms)
    product_center_atoms = np.array(product_center_atoms) + reactants.GetNumAtoms()

    node_center = torch.zeros([num_nodes,2],dtype = torch.uint8)
    node_center[reactant_center_atoms,0] = 1
    node_center[product_center_atoms,1] = 1

    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    message_graph.ndata['attribute'] = node_attribute.to(torch.uint8)
    message_graph.ndata['center'] = node_center.to(torch.uint8)
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
    
    offset = 0
    edges = []
    for index,molecule in enumerate(list(reaction.GetReactants())):
        edges += [(i + offset,num_atom_nodes + index) for i in range(molecule.GetNumAtoms())]
        offset += molecule.GetNumAtoms()
    edges = np.array(edges)
    aggregate_edge_attribute = np.zeros_like(edges)
    disperse_edge_attribute = np.zeros_like(edges)
    aggregate_edge_attribute[:,0] = 1
    disperse_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([aggregate_edge_attribute,disperse_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    reactant_molecule_message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reactant_molecule_message_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    edges = []
    for index,molecule in enumerate(list(reaction.GetProducts())):
        edges += [(i + offset,num_atom_nodes + len(reaction.GetReactants()) + index) for i in range(molecule.GetNumAtoms())]
        offset += molecule.GetNumAtoms()
    edges = np.array(edges)
    aggregate_edge_attribute = np.zeros_like(edges)
    disperse_edge_attribute = np.zeros_like(edges)
    aggregate_edge_attribute[:,0] = 1
    disperse_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([aggregate_edge_attribute,disperse_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    product_molecule_message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    product_molecule_message_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

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

    forward_edges = []
    backward_edges = []
    for reactant_molecule_index in range(len(reaction.GetReactants())):
        for product_molecule_index in range(len(reaction.GetProducts())):
            forward_edges.append((reactant_molecule_index + num_atom_nodes, product_molecule_index + num_atom_nodes + len(reaction.GetReactants())))
            backward_edges.append((product_molecule_index + num_atom_nodes + len(reaction.GetReactants()), reactant_molecule_index + num_atom_nodes))
    edges = forward_edges + backward_edges
    for reactant_molecule_index_1 in range(len(reaction.GetReactants())):
        for reactant_molecule_index_2 in range(len(reaction.GetReactants())):
            if reactant_molecule_index_1 != reactant_molecule_index_2:
                edges.append((reactant_molecule_index_1 + num_atom_nodes,reactant_molecule_index_2 + num_atom_nodes))
    for product_molecule_index_1 in range(len(reaction.GetProducts())):
        for product_molecule_index_2 in range(len(reaction.GetProducts())):
            if product_molecule_index_1 != product_molecule_index_2:
                edges.append((product_molecule_index_1 + num_atom_nodes + len(reaction.GetReactants()),product_molecule_index_2+ num_atom_nodes + len(reaction.GetReactants())))

    edge_attribute = torch.zeros([len(edges),4])
    edge_attribute[:len(forward_edges),0] = 1
    edge_attribute[len(forward_edges):2*len(forward_edges),1] = 1
    edge_attribute[2*len(forward_edges):2*len(forward_edges) + len(reaction.GetReactants())**2 - len(reaction.GetReactants()),2] = 1
    edge_attribute[2*len(forward_edges) + len(reaction.GetReactants())**2 - len(reaction.GetReactants()):,3] = 1
    edges = np.array(edges)
    src = edges[:,0]
    dst = edges[:,1]
    reaction_molecule_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reaction_molecule_message_passing_graph.edata['attribute'] = edge_attribute.to(torch.uint8)

    fingerprint = reactant_fingerprint + product_fingerprint + reactant_molecule_fingerprint + product_molecule_fingerprint
    
    return {"message_graph":message_graph,
            "reactant_message_passing_graph":reactant_message_passing_graph,
            "product_message_passing_graph":product_message_passing_graph,
            "reactant_molecule_message_graph":reactant_molecule_message_graph,
            "product_molecule_message_graph":product_molecule_message_graph,
            "reaction_atom_message_passing_graph":reaction_atom_message_passing_graph,
            "reaction_molecule_message_passing_graph":reaction_molecule_message_passing_graph,
            "fingerprint":fingerprint}

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = int, default = 0)
parser.add_argument("--dataset_dir", type = str, default = "")
parser.add_argument("--dim_hidden_features", type = int, default = 64)
parser.add_argument("--dim_hidden", type = int, default = 1024)
parser.add_argument("--message_passing_step", type = int, default = 3)
parser.add_argument("--pooling_step", type = int, default = 3)
parser.add_argument("--num_layers_pooling", type = int, default = 1)
parser.add_argument("--dim_hidden_regression", type = int, default = 512)
parser.add_argument("--dropout", type = float, default = 0.1)
parser.add_argument("--reactant",type=str,default="")
parser.add_argument("--reagent",type=str,default="")
parser.add_argument("--product",type=str,default="")
parser.add_argument("--workdir",type=str,default="")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


rxnmapper = RXNMapper()

reactant = args.reactant
reagent = args.reagent
product = args.product
smiles = f"{reactant}>>{product}"
smiles = rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
reactant,product = smiles.split(">>")
if reagent:
    reactant = f"{reactant}.{reagent}"
mapped_smiles = f"{reactant}>>{product}"

input_dict = get_reaction_graph(mapped_smiles)
input_dict = {key: [value] for key,value in input_dict.items()}
input_dict["message_graph"] = dgl.batch(input_dict["message_graph"])
input_dict["reactant_message_passing_graph"] = dgl.batch(input_dict["reactant_message_passing_graph"])
input_dict["product_message_passing_graph"] = dgl.batch(input_dict["product_message_passing_graph"])
input_dict["reactant_molecule_message_graph"] = dgl.batch(input_dict["reactant_molecule_message_graph"])
input_dict["product_molecule_message_graph"] = dgl.batch(input_dict["product_molecule_message_graph"])
input_dict["reaction_atom_message_passing_graph"] = dgl.batch(input_dict["reaction_atom_message_passing_graph"])
input_dict["reaction_molecule_message_passing_graph"] = dgl.batch(input_dict["reaction_molecule_message_passing_graph"])
input_dict["fingerprint"] = [fingerprint for fingerprints in input_dict["fingerprint"] for fingerprint in fingerprints]

reactants,products = mapped_smiles.split(">>")
reactants = reactants.split(".")
products = products.split(".")
reactants = [Chem.MolFromSmiles(reactant) for reactant in reactants]
products = [Chem.MolFromSmiles(product) for product in products]
molecules = reactants + products
num_node_batch = [molecule.GetNumAtoms() for molecule in molecules]
num_node_batch = torch.tensor(num_node_batch)

num_reactant_batch = [len(reactants)]
num_product_batch = [len(products)]
select_reactant = [1]* len(reactants) +  [0] * len(products)

num_reactant_batch = torch.tensor(num_reactant_batch)
num_product_batch = torch.tensor(num_product_batch)
select_reactant = torch.tensor(select_reactant).bool()
molecule_aggregation_graph = dgl.graph(([],[]),num_nodes=num_node_batch.sum())
molecule_aggregation_graph.set_batch_num_nodes(num_node_batch)

message_graph = input_dict["message_graph"]
reactant_message_passing_graph = input_dict["reactant_message_passing_graph"]
product_message_passing_graph = input_dict["product_message_passing_graph"]
reaction_atom_message_passing_graph = input_dict["reaction_atom_message_passing_graph"]

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

reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5])
reactant_atom_padding[:,0] = 1
reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

product_atom_padding = torch.zeros([product_atom_num_edge,5])
product_atom_padding[:,1] = 1
product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2])
reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1])
reaction_atom_edge_attribute = torch.cat([reaction_atom_padding_left,reaction_atom_edge_attribute,reaction_atom_padding_right],dim = -1)

reactant_message_passing_graph.edata["attribute"] = reactant_atom_edge_attribute
product_message_passing_graph.edata["attribute"] = product_atom_edge_attribute
reaction_atom_message_passing_graph.edata["attribute"] = reaction_atom_edge_attribute

message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reaction_atom_message_passing_graph])

node_type = message_graph.ndata["type"]
select_reactant_atoms = node_type[:,0] == 1
select_product_atoms = node_type[:,1] == 1
select_reactant_molecule = node_type[:,2] == 1
select_product_molecule = node_type[:,3] == 1
select_atoms = select_reactant_atoms | select_product_atoms
select_molecule = select_reactant_molecule | select_product_molecule

device = node_type.device

def get_num_node_each_batch(vector):
    count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
    count_list = torch.tensor(count_list).to(device)
    return count_list

num_atoms_each_batch = get_num_node_each_batch(select_atoms)

indices = torch.arange(node_type.size(0)).to(node_type.device)
remove_indices = indices[select_molecule]

message_graph.remove_nodes(remove_indices)
message_passing_graph.remove_nodes(remove_indices)
message_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_edges(num_edges_each_batch)

message_graph.ndata["attribute"] = torch.cat([message_graph.ndata["attribute"],message_graph.ndata["type"]],dim = -1)

inputs = {}
inputs["message_graph"] = message_graph
inputs["message_passing_graph"] = message_passing_graph
inputs["molecule_aggregation_graph"] = molecule_aggregation_graph
inputs["num_reactant_batch"] = num_reactant_batch
inputs["num_product_batch"] = num_product_batch
inputs["select_reactant"] = select_reactant

message_graph = inputs["message_graph"].to("cuda")
message_passing_graph = inputs["message_passing_graph"].to("cuda")
molecule_aggregation_graph = inputs["molecule_aggregation_graph"].to("cuda")
num_reactant_batch = inputs["num_reactant_batch"].cuda()
num_product_batch = inputs["num_product_batch"].cuda()
select_reactant = inputs["select_reactant"].cuda()
message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
inputs = {"message_graph":message_graph,
        "message_passing_graph":message_passing_graph,
        "molecule_aggregation_graph":molecule_aggregation_graph,
        "num_reactant_batch":num_reactant_batch,
        "num_product_batch":num_product_batch,
        "select_reactant":select_reactant}
    
mean = 0.568238250428795
std = 0.2665796062597878

network = Network(
    dim_hidden_features = args.dim_hidden_features,
    dim_hidden = args.dim_hidden,
    message_passing_step = args.message_passing_step,
    pooling_step = args.pooling_step,
    num_layers_pooling = args.num_layers_pooling,
    dim_hidden_regression = args.dim_hidden_regression,
    dropout = args.dropout
).cuda()
checkpoint = torch.load("uspto_subgram.pth")
network.load_state_dict(checkpoint)

test_mean = []
test_var = []

mean_list = []
var_list = []

with torch.no_grad():
    for _ in range(30):
        pred_mean, pred_logvar = network(inputs)
        mean_list.append(pred_mean.cpu().numpy())
        var_list.append(np.exp(pred_logvar.cpu().numpy()))

test_mean.append(np.array(mean_list).transpose())
test_var.append(np.array(var_list).transpose())

test_mean = np.vstack(test_mean) * std + mean
test_var = np.vstack(test_var) * std ** 2

prediction = float(np.mean(test_mean, 1)[0])
epistemic = float(np.var(test_mean, 1)[0])
aleatoric = float(np.mean(test_var, 1)[0])

print(prediction)
print(epistemic)
print(aleatoric)

result = {"yield":prediction,"variance":epistemic+aleatoric}
workdir = args.workdir
with open(f"{workdir}/uspto_subgram.json","w") as f:
    json.dump(result,f)