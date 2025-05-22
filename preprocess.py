import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures,AllChem
from dgl import graph
import warnings
import torch
import os
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle as pkl
from copy import deepcopy
import argparse
import pandas as pd
from rxnmapper import RXNMapper
from tqdm import tqdm
import dgl
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
atom_list = ['Ag','Al','As','B','Bi',
             'Br','C','Cl','Co','Cr',
             'Cu','F','Ge','H','I',
             'In','K','Li','Mg','Mo',
             'N','Na','O','P','Pd',
             'S','Sb','Se','Si','Sn',
             'Te','Zn','Os','Ti','Xe',
             'Ga','Ca','Zr','Gd','Rh',
             'Rb','Ba','Ce','La','Tb', 
             'V','Mn','Sm','W','Ru', 
             'Pt','Tl','Ni','Pb','Cd', 
             'Y','U','Hf','Fe','Hg','Au']

charge_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
degree_list = [1, 2, 3, 4, 5, 6, 7, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

def get_conformer(mol):
    mol = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, useRandomCoords = True, maxAttempts = 10) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    mol = AllChem.RemoveHs(mol)
    return mol

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

def get_atom_3d_info(mol):
    node_positions = torch.tensor(mol.GetConformer().GetPositions())
    return node_positions

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
    reactants,products = smiles.split(">>")
    reactants = Chem.MolFromSmiles(reactants)
    products = Chem.MolFromSmiles(products)

    if reactants is None or products is None:
        raise Exception("Invalid SMILES string")

    reactants = get_conformer(reactants)
    products = get_conformer(products)

    if reactants is None or products is None:
        raise Exception("Conformer generation failed")

    reaction = rdChemReactions.ReactionFromSmarts(smiles,useSmiles=True)
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

    reactant_positions = get_atom_3d_info(reactants)
    product_positions = get_atom_3d_info(products)
    
    node_attribute_padding = torch.stack([torch.zeros_like(reactant_attribute[0]) ]* num_molecule_nodes)
    node_positions_padding = torch.stack([torch.zeros_like(reactant_positions[0]) ]* num_molecule_nodes)

    reactant_molecule_fingerprint = get_molecule_fingerprint(reaction.GetReactants())
    product_molecule_fingerprint = get_molecule_fingerprint(reaction.GetProducts())

    node_attribute = torch.cat([reactant_attribute,product_attribute,node_attribute_padding],dim=0)
    node_positions = torch.cat([reactant_positions,product_positions,node_positions_padding],dim=0)

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
    message_graph = graph((src, dst), num_nodes = num_nodes)
    message_graph.ndata['attribute'] = node_attribute.to(torch.uint8)
    message_graph.ndata['position'] = node_positions.to(torch.float32)
    message_graph.ndata['center'] = node_center.to(torch.uint8)
    message_graph.ndata['type'] = node_type.to(torch.uint8)
    
    reactant_bond_attribute = get_atom_bond_attribute(reactants)
    reactant_bond_attribute = torch.cat([reactant_bond_attribute,reactant_bond_attribute],dim=0)
    reactant_bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in reactants.GetBonds()], dtype=int)
    src = np.hstack([reactant_bonds[:,0], reactant_bonds[:,1]])
    dst = np.hstack([reactant_bonds[:,1], reactant_bonds[:,0]])
    reactant_message_passing_graph = graph((src, dst), num_nodes = num_nodes)
    reactant_message_passing_graph.edata['attribute'] = reactant_bond_attribute.to(torch.uint8)

    product_bond_attribute = get_atom_bond_attribute(products)
    product_bond_attribute = torch.cat([product_bond_attribute,product_bond_attribute],dim=0)
    product_bonds = np.array([[b.GetBeginAtomIdx() + reactants.GetNumAtoms(), b.GetEndAtomIdx() + reactants.GetNumAtoms()] for b in products.GetBonds()], dtype=int)
    src = np.hstack([product_bonds[:,0], product_bonds[:,1]])
    dst = np.hstack([product_bonds[:,1], product_bonds[:,0]])
    product_message_passing_graph = graph((src, dst), num_nodes = num_nodes)
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
    reactant_molecule_message_graph = graph((src, dst), num_nodes = num_nodes)
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
    product_molecule_message_graph = graph((src, dst), num_nodes = num_nodes)
    product_molecule_message_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    assert 0 not in product_map_index, "Product map index should not contain 0"
    product_map_index = np.array(product_map_index) - 1
    product_index = np.arange(0,len(product_map_index)) + reactants.GetNumAtoms()
    edges = np.concatenate([product_map_index[:,np.newaxis],product_index[:,np.newaxis]],axis=-1)
    forward_edge_attribute = np.zeros_like(edges)
    backward_edge_attribute = np.zeros_like(edges)
    forward_edge_attribute[:,0] = 1
    backward_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([forward_edge_attribute,backward_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    reaction_atom_message_passing_graph = graph((src, dst), num_nodes = num_nodes)
    reaction_atom_message_passing_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8) # Construct Reaction Edges.

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
    reaction_molecule_message_passing_graph = graph((src, dst), num_nodes = num_nodes)
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
        current_batch_num_edges = len(new_edges) -last_num_edges
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

parser = argparse.ArgumentParser(description='hyperparameters')
parser.add_argument('--start', type=int,default=0)
parser.add_argument('--datas', type=str,default="train")
parser.add_argument('--device', type=str,default="0")
parser.add_argument('--split_num', type=int,default=64)
parser.add_argument('--batch_size', type=int,default=32)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
split_num = args.split_num
batch_size = args.batch_size
     
class ReactionGraphPreprocessor:
    def __init__(self,
                 source_dir,
                 target_dir,
                 split_num,
                 batch_size,
                 datas,
                 start,
                 dataset_prefix):
        
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.split_num = split_num
        self.batch_size = batch_size

        self.dataset_prefix = dataset_prefix
        self.start = start
        self.datas = datas
        self.rxnmapper = RXNMapper()

        self.compound_dictionary = f"{source_dir}/USPTO_Condition_Dictionary.pkl"
        self.reaction_dictionary = {
            "train": f"{source_dir}/USPTO_Condition_Reaction_Dictionary_Train.pkl",
            "test": f"{source_dir}/USPTO_Condition_Reaction_Dictionary_Test.pkl",
            "val": f"{source_dir}/USPTO_Condition_Reaction_Dictionary_Val.pkl",
        }
        self.dataset = {
            "train": f"{source_dir}/USPTO_Condition_Train.csv",
            "test": f"{source_dir}/USPTO_Condition_Test.csv",
            "val": f"{source_dir}/USPTO_Condition_Val.csv",
        }
        self.solvents = f"{source_dir}/USPTO_Condition_Solvent.pkl"
        self.catalysts = f"{source_dir}/USPTO_Condition_Catalyst.pkl"
        self.reagents = f"{source_dir}/USPTO_Condition_Reagent.pkl"

        with open(self.compound_dictionary,"rb") as f:
            self.compound_dictionary = pkl.load(f)
        for key in self.reaction_dictionary:
            with open(self.reaction_dictionary[key],"rb") as f:
                self.reaction_dictionary[key] = pkl.load(f)
        for key in self.dataset:
            self.dataset[key] = pd.read_csv(self.dataset[key])
        
        with open(self.solvents,"rb") as f:
            self.solvents = np.array(pkl.load(f))
        with open(self.catalysts,"rb") as f:
            self.catalysts = np.array(pkl.load(f))
        with open(self.reagents,"rb") as f:
            self.reagents = np.array(pkl.load(f))

    def encode_condition(self,row):
        catalyst1 = str(row["catalyst1"]) == self.catalysts
        solvent1 = str(row["solvent1"]) == self.solvents
        solvent2 = str(row["solvent2"]) == self.solvents
        reagent1 = str(row["reagent1"]) == self.reagents
        reagent2 = str(row["reagent2"]) == self.reagents
        encoding = np.concatenate([catalyst1,solvent1,solvent2,reagent1,reagent2]).astype(np.byte)
        return encoding

    def preprocess(self):
        dataset = self.dataset[self.datas]
        split_size = len(dataset) // self.split_num + 1
        dataset = dataset[self.start * split_size:(self.start + 1) * split_size]
        progress = tqdm(dataset.iterrows(),total=len(dataset))

        batches = []
        error_list = []
        key_list = []
        input_list = []
        output_list = []
        for index,row in progress:
            smiles = row["canonical_rxn"]
            try:
                mapped_smiles = self.rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
                return_dict = get_reaction_graph(mapped_smiles)
                if return_dict is None:
                    raise Exception("Empty return dict!")
                key_list.append(smiles)
                input_list.append(return_dict)
                output_list.append(self.encode_condition(row))
            except Exception as e:
                error_list.append({"smiles":smiles,"error":str(e)})
                print(e)
                continue
            if index % self.batch_size == self.batch_size - 1:
                input_dict = {key:[item[key] for item in input_list] for key in input_list[0]}
                input_dict["message_graph"] = dgl.batch(input_dict["message_graph"])
                input_dict["reactant_message_passing_graph"] = dgl.batch(input_dict["reactant_message_passing_graph"])
                input_dict["product_message_passing_graph"] = dgl.batch(input_dict["product_message_passing_graph"])
                input_dict["reactant_molecule_message_graph"] = dgl.batch(input_dict["reactant_molecule_message_graph"])
                input_dict["product_molecule_message_graph"] = dgl.batch(input_dict["product_molecule_message_graph"])
                input_dict["reaction_atom_message_passing_graph"] = dgl.batch(input_dict["reaction_atom_message_passing_graph"])
                input_dict["reaction_molecule_message_passing_graph"] = dgl.batch(input_dict["reaction_molecule_message_passing_graph"])
                input_dict["fingerprint"] = [fingerprint for fingerprints in input_dict["fingerprint"] for fingerprint in fingerprints]
                batch = {}
                batch["keys"] = deepcopy(key_list)
                batch["inputs"] = input_dict
                batch["outputs"] = np.array(output_list)
                batches.append(batch)
                key_list = []
                input_list = []
                output_list = []
        
        if len(key_list) > 0:
            input_dict = {key:[item[key] for item in input_list] for key in input_list[0]}
            input_dict["message_graph"] = dgl.batch(input_dict["message_graph"])
            input_dict["reactant_message_passing_graph"] = dgl.batch(input_dict["reactant_message_passing_graph"])
            input_dict["product_message_passing_graph"] = dgl.batch(input_dict["product_message_passing_graph"])
            input_dict["reactant_molecule_message_graph"] = dgl.batch(input_dict["reactant_molecule_message_graph"])
            input_dict["product_molecule_message_graph"] = dgl.batch(input_dict["product_molecule_message_graph"])
            input_dict["reaction_atom_message_passing_graph"] = dgl.batch(input_dict["reaction_atom_message_passing_graph"])
            input_dict["reaction_molecule_message_passing_graph"] = dgl.batch(input_dict["reaction_molecule_message_passing_graph"])
            input_dict["fingerprint"] = [fingerprint for fingerprints in input_dict["fingerprint"] for fingerprint in fingerprints]
            batch = {}
            batch["keys"] = deepcopy(key_list)
            batch["inputs"] = input_dict
            batch["outputs"] = np.array(output_list)
            batches.append(batch)
            key_list = []
            input_list = []
            output_list = []

        for batch in tqdm(dataset): # Construct Angular Edges.
            inputs = batch["inputs"]
            message_graph = inputs["message_graph"]
            reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
            product_message_passing_graph = inputs["product_message_passing_graph"]
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

            inputs["reactant_message_passing_graph"] = reactant_message_passing_graph
            inputs["product_message_passing_graph"] = product_message_passing_graph
            inputs["reactant_geometry_message_graph"] = reactant_geometry_message_graph
            inputs["product_geometry_message_graph"] = product_geometry_message_graph

            molecule_aggregation_graph = inputs["molecule_aggregation_graph"]
            fixed_molecule_aggregation_graph = dgl.graph([],num_nodes=molecule_aggregation_graph.batch_num_nodes().sum())
            fixed_molecule_aggregation_graph.set_batch_num_nodes(molecule_aggregation_graph.batch_num_nodes())
            fixed_molecule_aggregation_graph.set_batch_num_edges(molecule_aggregation_graph.batch_num_edges())
            inputs["molecule_aggregation_graph"] = fixed_molecule_aggregation_graph
            batch["inputs"] = inputs

        if not os.path.exists(f"{self.target_dir}"):
            os.mkdir(f"{self.target_dir}")
        index_name = self.start
        

        split_name = f"{self.dataset_prefix}{self.datas.capitalize()}{index_name}_{self.split_size}.pkl"
        with open(f"{self.target_dir}/{split_name}","wb") as f:
            pkl.dump(batches,f)

        error_log_name = f"{self.dataset_prefix}Error{self.datas.capitalize()}{index_name}_{self.split_size}.pkl"
        with open(f"{self.target_dir}/{error_log_name}","wb") as f:
            pkl.dump(error_list,f)
        print("Done!")

preprocessor = ReactionGraphPreprocessor(source_dir = "dataset",
                               target_dir = "dataset",
                               split_num=split_num,
                               batch_size=batch_size,
                               datas = args.datas,
                               start = args.start,
                               dataset_prefix="ReactionGraph")
preprocessor.preprocess()