import os
import random
import warnings
from argparse import ArgumentParser
import numpy as np

import torch
from torch import nn
from torch.nn import Module

import dgl
from dgl.nn.pytorch import NNConv

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures,AllChem,rdChemReactions

from rxnmapper import RXNMapper
from itertools import groupby

import json

from dgl.readout import sum_nodes, broadcast_nodes,softmax_nodes

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

attention_weight = None

class Set2Set(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        global attention_weight
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                if attention_weight is None:
                    attention_weight = alpha.cpu().numpy()
                
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = torch.cat([q, readout], dim=-1)

            return q_star

    def extra_repr(self):
        summary = 'n_iters={n_iters}'
        return summary.format(**self.__dict__)

class RBFLayer(torch.nn.Module):
    def __init__(self, dim):
        super(RBFLayer, self).__init__()
        self.dim = dim
        self.centers = torch.nn.Parameter(torch.Tensor(dim))
        self.beta = torch.nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.centers, 0, 1)
        torch.nn.init.constant_(self.beta, 10)

    def forward(self, x):
        x = x.view(-1, 1)
        centers = self.centers.view(1, -1)
        diff = x - centers
        dist_sq = torch.square(diff)
        out = torch.exp(-self.beta * dist_sq)
        return out

class TypeNN(Module):
    def __init__(self,
                 dim_node_attribute = 117,
                 dim_edge_attribute = 8,
                 dim_edge_length = 8,
                 dim_hidden_features = 64,
                 dim_hidden = 4096,
                 message_passing_step = 4,
                 pooling_step = 3,
                 num_layers_pooling = 2,
                 dim_type = 12):
        
        super(TypeNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(
            nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
        self.rbf = RBFLayer(dim_edge_length)
        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
    
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)

        self.pooling = Set2Set(input_dim = dim_hidden_features * 2,
                               n_iters = pooling_step,
                               n_layers = num_layers_pooling)

        self.sparsify = nn.Sequential(
            nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.PReLU(),
            nn.Linear(dim_hidden, dim_type)
        )

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]

        node_attribute = message_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        
        edge_attribute = message_passing_graph.edata["attribute"]
        edge_length = self.rbf(message_passing_graph.edata["length"])
        edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(message_passing_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        reaction_features = self.pooling(message_passing_graph, node_aggregation)
        output = self.sparsify(reaction_features)
        return output

class ReactionTypeModel:
    def __init__(self,
                 dataset_train = None,
                 dataset_val = None,
                 dataset_test = None,
                 model_dir = "",
                 device = "cuda",
                 max_gradient = 1e2,
                 parameters_for_model = {},
                 parameters_for_optimizer = {"lr":0.0005, 
                                             "weight_decay":1e-10},
                 parameters_for_scheduler = {"mode":"min", 
                                             "factor":0.1, 
                                             "patience":20, 
                                             "min_lr":1e-7, 
                                             "verbose":True},
                 accumulation_steps = 4):
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.max_gradient = max_gradient
        self.device = device
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.accmulation_steps = accumulation_steps
        
        self.model = TypeNN(**parameters_for_model).to(device)

    @torch.no_grad()
    def inference(self,inputs):
        self.model.eval()
        return self.model(inputs).argmax(-1)

    def load(self,filename):
        state_dict = torch.load(f"{self.model_dir}/{filename}")
        self.model.load_state_dict(state_dict)

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
atom_list = ["H","Er","Th","Lu","Ga",
            "Mg","Au","Sr","Ru","Cd",
            "Ir","At","Y","Br","V",
            "Pb","S","Pd","Ca","As",
            "K","In","Al","Be","Tb",
            "Ce","Pt","Bi","Ni","P",
            "U","Gd","Tc","Dy","C",
            "Se","N","Eu","Fe","Rh",
            "Sm","H","Sg","Zn","Mo",
            "Sn","Sc","Cu","Ge","I",
            "O","Ag","Nd","Mn","Sb",
            "Cl","Zr","Hf","Cs","Hg",
            "F","Ti","Re","Ta","Os",
            "Te","Cr","Li","Pr","Xe",
            "W","Co","Yb","B","Na",
            "Tl","Si"]
charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
degree_list = [1, 2, 3, 4, 5, 6, 7, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'UNSPECIFIED']

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

def get_3d_for_one(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    coordinates = None
    if AllChem.EmbedMolecule(mol, useRandomCoords = True, maxAttempts = 100) != -1:
        if AllChem.MMFFOptimizeMolecule(mol) != -1:
            mol = AllChem.RemoveHs(mol)
            coordinates = np.array(mol.GetConformer().GetPositions())
        elif AllChem.UFFOptimizeMolecule(mol) != -1:
            mol = AllChem.RemoveHs(mol)
            coordinates = np.array(mol.GetConformer().GetPositions())
        else:
            mol = AllChem.RemoveHs(mol)
            coordinates = np.array(mol.GetConformer().GetPositions())
    else:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        coordinates = np.array(mol.GetConformer().GetPositions())
    return coordinates

def initialize_stereo_info(mol):
    for element in Chem.FindPotentialStereo(mol):
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': 
            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': 
            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    return mol

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

def get_atom_bond_attribute(mol):
    bond_feature1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
    bond_feature2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
    bond_feature3 = np.array([get_stereochemistry_info(b) for b in mol.GetBonds()], dtype = bool)
    edge_attribute = np.hstack([bond_feature1, bond_feature2, bond_feature3])
    edge_attribute = torch.tensor(edge_attribute)
    return edge_attribute

reactant_mol_blocks = []
product_mol_blocks = []
mol_blocks = []
def get_atom_3d_info_compute(smiles):
    molecules = smiles.split(".")
    node_positions_list = []
    for molecule in molecules:
        mol = Chem.MolFromSmiles(molecule)
        node_positions = get_3d_for_one(molecule)
        conf = Chem.Conformer(mol.GetNumAtoms())
    
        for i, pos in enumerate(node_positions):
            conf.SetAtomPosition(i, pos)

        mol.AddConformer(conf)
        mol_block = Chem.MolToMolBlock(mol)
        mol_blocks.append(mol_block)

        node_positions = torch.tensor(node_positions)
        node_positions_list.append(node_positions)
    node_positions = torch.cat(node_positions_list,dim=0)
    return node_positions

def get_reaction_type_graph_inference(mapped_smiles = ""):
    global mol_blocks
    global reactant_mol_blocks
    global product_mol_blocks

    reactants_smiles,products_smiles = mapped_smiles.split(">>")
    num_of_reactants_atoms = len(Chem.MolFromSmiles(reactants_smiles).GetAtoms())
    reactants = Chem.MolFromSmiles(reactants_smiles)
    products = Chem.MolFromSmiles(products_smiles)

    if reactants is None or products is None:
        raise Exception("Invalid SMILES string")
    mapped_smiles = f"{reactants_smiles}>>{products_smiles}"

    reaction = rdChemReactions.ReactionFromSmarts(mapped_smiles)
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

    reactant_positions = get_atom_3d_info_compute(reactants_smiles)
    reactant_mol_blocks = mol_blocks
    mol_blocks = []
    product_positions = get_atom_3d_info_compute(products_smiles)
    product_mol_blocks = mol_blocks
    mol_blocks = []

    assert reactant_positions.shape[0] == reactants.GetNumAtoms(), "Not match position size"
    assert product_positions.shape[0] == products.GetNumAtoms(), "Not match position size"
    
    node_attribute_padding = torch.stack([torch.zeros_like(reactant_attribute[0]) ]* num_molecule_nodes)
    node_positions_padding = torch.stack([torch.zeros_like(reactant_positions[0]) ]* num_molecule_nodes)

    node_attribute = torch.cat([reactant_attribute,product_attribute,node_attribute_padding],dim=0)
    node_positions = torch.cat([reactant_positions,product_positions,node_positions_padding],dim=0)

    node_type = torch.zeros([num_nodes,5], dtype = torch.uint8)
    node_type[:num_of_reactants_atoms,0] = 1
    node_type[num_of_reactants_atoms:reactants.GetNumAtoms(),4] = 1
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
    message_graph.ndata['position'] = node_positions.to(torch.float32)
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

    return {"message_graph":message_graph,
            "reactant_message_passing_graph":reactant_message_passing_graph,
            "product_message_passing_graph":product_message_passing_graph,
            "reactant_molecule_message_graph":reactant_molecule_message_graph,
            "product_molecule_message_graph":product_molecule_message_graph,
            "reaction_atom_message_passing_graph":reaction_atom_message_passing_graph,
            "reaction_molecule_message_passing_graph":reaction_molecule_message_passing_graph}

def construct(g):
    num_nodes = g.num_nodes()
    batch_num_nodes = g.batch_num_nodes()
    base = 0
    batch_num_edges = []
    new_edges = set()
    last_num_edges = 0
    for current_batch_num_nodes in batch_num_nodes:
        for offset in range(current_batch_num_nodes):
            i = base + offset
            predecessors = g.predecessors(i)
            for j in predecessors:
                for k in predecessors:
                    if j!=k:
                        new_edges.add((j.item(),k.item()))
        base += current_batch_num_nodes
        current_batch_num_edges = len(new_edges) - last_num_edges
        last_num_edges = len(new_edges)
        batch_num_edges.append(current_batch_num_edges)

    batch_num_edges = torch.tensor(batch_num_edges)
    src,dst = list(zip(*new_edges))
    src = np.array(src)
    dst = np.array(dst)
    new_g = dgl.graph((src,dst),num_nodes=num_nodes)
    new_g.set_batch_num_nodes(batch_num_nodes)
    new_g.set_batch_num_edges(batch_num_edges)
    return new_g

parser = ArgumentParser(description='hyperparameters')
parser.add_argument('--seed', type=float,default=123)
parser.add_argument('--device', type=str,default="3")
parser.add_argument('--dataset', type=str,default="../dataset")
parser.add_argument('--model_dir', type=str,default="./")
parser.add_argument('--ablation', type=str, default="none")
parser.add_argument("--reactant",type=str,default="CO")
parser.add_argument("--reagent",type=str,default="")
parser.add_argument("--product",type=str,default="CO")
parser.add_argument("--workdir",type=str,default="workdir")

parser.add_argument("--epoches", type=int, default=100)

parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=1e-10)
parser.add_argument("--learning_rate_schedule", type=str,default="min") 
parser.add_argument("--learning_rate_factor",type=float,default=0.1)
parser.add_argument("--learning_rate_patience",type=int,default=5)
parser.add_argument("--min_learning_rate",type=float,default=1e-8)
parser.add_argument("--learning_rate_verbose",type=bool,default=True)

parser.add_argument("--dim_hidden", type=int, default=4096)
parser.add_argument("--dim_node_attribute", type=int,default = 132)
parser.add_argument("--dim_edge_attribute", type=int,default = 14)
parser.add_argument("--dim_edge_length", type=int,default = 16)
parser.add_argument("--dim_hidden_features", type=int,default = 200)
parser.add_argument("--message_passing_step", type=int,default = 3)
parser.add_argument("--pooling_step", type=int,default = 2)
parser.add_argument("--num_layers_pooling", type=int,default = 2)

parser.add_argument("--save_delta", type=int,default = 1)
parser.add_argument("--accumulation_steps", type=int,default = 1)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
dgl.random.seed(seed)
torch.backends.cudnn.deterministic = True

model = ReactionTypeModel(dataset_train = None,
                          dataset_test = None,
                          dataset_val = None,
                          model_dir = args.model_dir,
                          parameters_for_model = {
                              "dim_hidden":args.dim_hidden,
                              "dim_node_attribute":args.dim_node_attribute,
                              "dim_edge_attribute":args.dim_edge_attribute,
                              "dim_edge_length":args.dim_edge_length,
                              "dim_hidden_features":args.dim_hidden_features,
                              "message_passing_step":args.message_passing_step,
                              "pooling_step":args.pooling_step,
                              "num_layers_pooling":args.num_layers_pooling,
                              "dim_type":12},
                          parameters_for_optimizer = {
                              "lr":args.learning_rate, 
                              "weight_decay":args.weight_decay},
                          parameters_for_scheduler = {
                              "mode":args.learning_rate_schedule, 
                              "factor":args.learning_rate_factor, 
                              "patience":args.learning_rate_patience, 
                              "min_lr":args.min_learning_rate, 
                              "verbose":args.learning_rate_verbose},
                          accumulation_steps=args.accumulation_steps)

mapper = RXNMapper()

model.load("pistachio_type.ckpt")
dataset = []
graphs = []

reactant = args.reactant
reagent = args.reagent
product = args.product
smiles = reactant + ">>" + product
smiles = mapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
reactant,product = smiles.split(">>")
if reagent:
    reactant = reactant + "." + reagent
smiles = reactant + ">>" + product

workdir = args.workdir
with open(f"{workdir}/mapped_smiles.json","w") as f:
    json.dump({"smiles":smiles},f)

inputs = get_reaction_type_graph_inference(smiles)

with open(f"{workdir}/coordinates.json","w") as f:
    json.dump({
        "reactant":reactant_mol_blocks,
        "product":product_mol_blocks
    },f)

message_graph = dgl.batch([inputs["message_graph"]])
reactant_message_passing_graph = dgl.batch([inputs["reactant_message_passing_graph"]])
product_message_passing_graph = dgl.batch([inputs["product_message_passing_graph"]])
reaction_atom_message_passing_graph = dgl.batch([inputs["reaction_atom_message_passing_graph"]])
reactant_geometry_message_graph = construct(reactant_message_passing_graph)
product_geometry_message_graph = construct(product_message_passing_graph)
positions = message_graph.ndata["position"]

def get_direction(graph):
    src = graph.edges()[0]
    dst = graph.edges()[1]
    directions = positions[dst] - positions[src]
    graph.edata["direction"] = directions

get_direction(reactant_message_passing_graph)
get_direction(product_message_passing_graph)
get_direction(reactant_geometry_message_graph)
get_direction(product_geometry_message_graph)

message_graph.ndata["attribute"] = torch.cat([message_graph.ndata["attribute"],message_graph.ndata["type"]],dim = -1)

num_edges_each_batch = 0
num_edges_each_batch += reactant_message_passing_graph.batch_num_edges()
num_edges_each_batch += product_message_passing_graph.batch_num_edges()
num_edges_each_batch += reactant_geometry_message_graph.batch_num_edges()
num_edges_each_batch += product_geometry_message_graph.batch_num_edges()
num_edges_each_batch += reaction_atom_message_passing_graph.batch_num_edges()

reactant_atom_edge_attribute = reactant_message_passing_graph.edata["attribute"]
product_atom_edge_attribute = product_message_passing_graph.edata["attribute"]
reaction_atom_edge_attribute = reaction_atom_message_passing_graph.edata["attribute"]

reactant_atom_num_edge = reactant_message_passing_graph.num_edges()
product_atom_num_edge = product_message_passing_graph.num_edges()
reactant_geometry_atom_num_edge = reactant_geometry_message_graph.num_edges()
product_geometry_atom_num_edge = product_geometry_message_graph.num_edges()
reaction_atom_num_edge = reaction_atom_message_passing_graph.num_edges()

dim_edge_attribute = reactant_atom_edge_attribute.size(1)

reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5]).float()
reactant_atom_padding[:,0] = 1
reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

product_atom_padding = torch.zeros([product_atom_num_edge,5]).float()
product_atom_padding[:,1] = 1
product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

reactant_geometry_atom_edge_attribute = torch.zeros([reactant_geometry_atom_num_edge,dim_edge_attribute + 5]).float()
reactant_geometry_atom_edge_attribute[:,-1] = 1

product_geometry_atom_edge_attribute = torch.zeros([product_geometry_atom_num_edge,dim_edge_attribute + 5]).float()
product_geometry_atom_edge_attribute[:,-1] = 1

reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2]).float()
reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1]).float()
reaction_atom_edge_attribute = torch.cat([reaction_atom_padding_left,reaction_atom_edge_attribute,reaction_atom_padding_right],dim = -1)

reactant_message_passing_graph.edata["attribute"] = reactant_atom_edge_attribute
product_message_passing_graph.edata["attribute"] = product_atom_edge_attribute
reactant_geometry_message_graph.edata["attribute"] = reactant_geometry_atom_edge_attribute
product_geometry_message_graph.edata["attribute"] = product_geometry_atom_edge_attribute
reaction_atom_message_passing_graph.edata["attribute"] = reaction_atom_edge_attribute

reactant_message_passing_graph.edata['length'] = reactant_message_passing_graph.edata['direction'].norm(dim=-1)
del reactant_message_passing_graph.edata['direction']
product_message_passing_graph.edata['length'] = product_message_passing_graph.edata['direction'].norm(dim=-1)
del product_message_passing_graph.edata['direction']
reactant_geometry_message_graph.edata['length'] = reactant_geometry_message_graph.edata['direction'].norm(dim=-1)
del reactant_geometry_message_graph.edata['direction']
product_geometry_message_graph.edata['length'] = product_geometry_message_graph.edata['direction'].norm(dim=-1)
del product_geometry_message_graph.edata['direction']
reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge]).float()

message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reactant_geometry_message_graph,product_geometry_message_graph,reaction_atom_message_passing_graph])

node_type = message_graph.ndata["type"]
select_reactant_atoms = (node_type[:,0] == 1) | (node_type[:,4] == 1)
select_product_atoms = node_type[:,1] == 1
select_reactant_molecule = node_type[:,2] == 1
select_product_molecule = node_type[:,3] == 1
select_atoms = select_reactant_atoms | select_product_atoms
select_molecule = select_reactant_molecule | select_product_molecule

def get_num_node_each_batch(vector):
    count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
    count_list = torch.tensor(count_list)
    return count_list

num_atoms_each_batch = get_num_node_each_batch(select_atoms)

indices = torch.arange(node_type.size(0))
remove_indices = indices[select_molecule]

message_graph.remove_nodes(remove_indices)
message_passing_graph.remove_nodes(remove_indices)
message_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_edges(num_edges_each_batch)

inputs = {}
inputs["message_graph"] = message_graph
inputs["message_passing_graph"] = message_passing_graph

message_graph = inputs["message_graph"].to("cuda")
message_passing_graph = inputs["message_passing_graph"].to("cuda")
message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
inputs = {"message_graph":message_graph,
        "message_passing_graph":message_passing_graph}
result = model.inference(inputs)
attention_weight = attention_weight.ravel().tolist()
with open(f"{workdir}/attention_weights.json","w") as f:
    json.dump({"weight":attention_weight},f)

type_map = ["Unrecognized",
"Heteroatom alkylation and arylation",
"Acylation and related processes",
"C-C bond formation",
"Heterocycle formation",
"Protections",
"Deprotections",
"Reductions",
"Oxidations",
"Functional group interconversion (FGI)",
"Functional group addition (FGA)",
"Resolutions",
]

with open(f"{workdir}/result.json","w") as f:
    json.dump({"type":type_map[int(result)]},f)
print(workdir)