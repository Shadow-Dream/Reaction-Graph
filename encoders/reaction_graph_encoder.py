import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures,AllChem,rdchem
import warnings
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle as pkl
from copy import deepcopy
import argparse
import pandas as pd
from rxnmapper import RXNMapper
from tqdm import tqdm
import dgl
from metadatas import DEFAULT_METADATA,ConfigLoader
import re
import utils
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

class ReactionGraphEncoder:
    def __init__(self,metadata = {},**kwargs):
        self.rxnmapper = RXNMapper()
        self.chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        config_loader = ConfigLoader(DEFAULT_METADATA,metadata,kwargs)
        config_loader.apply(self)

    def calculate_conformer(self,mol):
        mol = AllChem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, useRandomCoords = True, maxAttempts = self.calculate_3d_position_max_attempts) == -1:
            if self.must_have_3d_position:
                raise Exception("Failed to calculate the 3d conformer.")
            else:
                mol = AllChem.RemoveHs(mol)
                if AllChem.Compute2DCoords(mol) == -1:
                    zero_conformer = rdchem.Conformer(mol.GetNumAtoms())
                    for i in range(mol.GetNumAtoms()):
                        zero_conformer.SetAtomPosition(i, [0,0,0])
                    mol.AddConformer(zero_conformer)
        else:
            if AllChem.MMFFOptimizeMolecule(mol) == -1:
                AllChem.UFFOptimizeMolecule(mol)
            mol = AllChem.RemoveHs(mol)
        return mol

    def get_reaction_mapping(self,reactants,products):
        map_index = [0 for _ in range(len(products.GetAtoms()))]
        for reactant_atom in reactants.GetAtoms():
            map_number = reactant_atom.GetAtomMapNum()
            for product_atom in products.GetAtoms():
                if product_atom.GetAtomMapNum() == map_number:
                    map_index[product_atom.GetIdx()] = reactant_atom.GetIdx() + 1
                    break
        return map_index

    def get_donor_and_acceptor_info(self,mol):
        donor_info, acceptor_info = [], []
        for feature in self.chem_feature_factory.GetFeaturesForMol(mol):
            if feature.GetFamily() == 'Donor': donor_info.append(feature.GetAtomIds()[0])
            if feature.GetFamily() == 'Acceptor': acceptor_info.append(feature.GetAtomIds()[0])
        return donor_info, acceptor_info

    def get_chirality_info(self,atom):
        return ([(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
                if atom.HasProp('Chirality') 
                else [0, 0])
    
    def get_stereochemistry_info(self,bond):
        return ([(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')]
                if bond.HasProp('Stereochemistry') 
                else [0, 0]) 

    def initialize_stereo_info(self,mol):
        for element in Chem.FindPotentialStereo(mol):
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': 
                mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': 
                mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        return mol

    def get_atom_3d_info(self,mol):
        node_positions = torch.tensor(mol.GetConformer().GetPositions())
        return node_positions

    def get_atom_attribute(self,mol):
        mol = self.initialize_stereo_info(mol)
        donor_list, acceptor_list = self.get_donor_and_acceptor_info(mol) 
        atom_feature1 = np.eye(len(self.atom_list), dtype = bool)[[self.atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]][:,:self.atom_dim]
        atom_feature2 = np.eye(len(self.charge_list), dtype = bool)[[self.charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:self.charge_dim]
        atom_feature3 = np.eye(len(self.degree_list), dtype = bool)[[self.degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:self.degree_dim]
        atom_feature4 = np.eye(len(self.hybridization_list), dtype = bool)[[self.hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:self.hybridization_dim]
        atom_feature5 = np.eye(len(self.hydrogen_list), dtype = bool)[[self.hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:self.hydrogen_dim]
        atom_feature6 = np.eye(len(self.valence_list), dtype = bool)[[self.valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:self.valence_dim]
        atom_feature7 = np.array([[(j in donor_list), (j in acceptor_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_feature8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        atom_feature9 = np.array([[a.IsInRingSize(s) for s in self.ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_feature10 = np.array([self.get_chirality_info(a) for a in mol.GetAtoms()], dtype = bool)
        attribute = np.hstack([atom_feature1, atom_feature2, atom_feature3, atom_feature4, atom_feature5, 
                               atom_feature6, atom_feature7, atom_feature8, atom_feature9, atom_feature10])
        attribute = torch.tensor(attribute)
        return attribute

    def get_atom_bond_attribute(self,mol):
        bond_feature1 = np.eye(len(self.bond_list), dtype = bool)[[self.bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]][:,:self.bond_dim]
        bond_feature2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool).reshape(-1,2)
        bond_feature3 = np.array([self.get_stereochemistry_info(b) for b in mol.GetBonds()], dtype = bool).reshape(-1,2)
        edge_attribute = np.hstack([bond_feature1, bond_feature2, bond_feature3])
        edge_attribute = torch.tensor(edge_attribute)
        return edge_attribute
    
    def preprocess_reaction(self,smiles):
        if "|" in smiles:
            smiles = smiles.split(" ")[0]
        mapping_pattern = re.compile(r':(\d+)]')
        if not re.findall(mapping_pattern, smiles):
            smiles = self.rxnmapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']
            reactants,reagents,products = smiles.split(">")
            reactant_molecules = reactants.split(".")
            new_reactant_molecules = []
            new_reagent_molecules = []
            for molecule in reactant_molecules:
                if re.findall(mapping_pattern, molecule):
                    new_reactant_molecules.append(molecule)
                else:
                    new_reagent_molecules.append(molecule)
            reactants = ".".join(new_reactant_molecules)
            new_reagents = ".".join(new_reagent_molecules)
            if new_reagents and reagents:
                reagents = new_reagents + "." + reagents
            elif new_reagents:
                reagents = new_reagents
            smiles = reactants + ">" + reagents  + ">" + products

        reactants,reagents,products = smiles.split(">")
        if reagents:
            reactants = reagents + "." + reactants
            reagent_molecules = [Chem.MolFromSmiles(molecule) for molecule in reagents.split(".")]
            reagents = Chem.MolFromSmiles(reagents)
        else:
            reagent_molecules = []
            reagents = Chem.MolFromSmiles("")
        reactant_mappings = sorted(re.findall(mapping_pattern, reactants))
        product_mappings = sorted(re.findall(mapping_pattern, products))
        if reactant_mappings != product_mappings:
            raise Exception("The mapping of reaction contains errors.")
        
        reactant_molecules = reactants.split(".")
        product_molecules = products.split(".")
        reactant_molecules = [Chem.MolFromSmiles(molecule) for molecule in reactant_molecules]
        product_molecules = [Chem.MolFromSmiles(molecule) for molecule in product_molecules]
        reactants = Chem.MolFromSmiles(reactants)
        products = Chem.MolFromSmiles(products)
        if reactants is None or products is None:
            raise Exception("The reaction smiles is invalid.")

        return reactants,reagents,products,reactant_molecules,reagent_molecules,product_molecules

    def get_reaction_graph(self,smiles):
        graph_components = []
        (
            reactants,
            reagents,
            products,
            reactant_molecules,
            reagent_molecules,
            product_molecules
        ) = self.preprocess_reaction(smiles)

        atom_per_molecule = torch.tensor([
            molecule.GetNumAtoms() 
            for molecule 
            in reactant_molecules + product_molecules
        ]).long()

        molecule_per_reaction = torch.tensor([
            len(reagent_molecules),
            len(reactant_molecules) - len(reagent_molecules),
            len(product_molecules)
        ]).long()

        if self.use_3d:
            reactants = self.calculate_conformer(reactants)
            products = self.calculate_conformer(products)

        map_index = self.get_reaction_mapping(reactants,products)
        
        num_nodes = reactants.GetNumAtoms() + products.GetNumAtoms()

        reactant_attribute = self.get_atom_attribute(reactants)
        product_attribute = self.get_atom_attribute(products)
        node_attribute = torch.cat([reactant_attribute,product_attribute])
        if self.use_3d:
            reactant_position = self.get_atom_3d_info(reactants)
            product_position = self.get_atom_3d_info(products)
            node_position = torch.cat([reactant_position,product_position])

        node_type = torch.zeros([num_nodes,5])
        node_type[:reagents.GetNumAtoms(),4] = 1
        node_type[reagents.GetNumAtoms():reactants.GetNumAtoms(),0] = 1
        node_type[reactants.GetNumAtoms():,1] = 1

        reactant_bond_attribute = self.get_atom_bond_attribute(reactants)
        reactant_bond_type = torch.zeros([reactants.GetNumBonds(),5])
        reactant_bond_type[:,0] = 1
        reactant_bond_attribute = torch.cat([reactant_bond_attribute,reactant_bond_type],dim = -1)
        product_bond_attribute = self.get_atom_bond_attribute(products)
        product_bond_type = torch.zeros([products.GetNumBonds(),5])
        product_bond_type[:,1] = 1
        product_bond_attribute = torch.cat([product_bond_attribute,product_bond_type],dim = -1)
        bond_attribute = torch.cat([reactant_bond_attribute,product_bond_attribute,reactant_bond_attribute,product_bond_attribute])
        bonds = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in list(reactants.GetBonds()) + list(products.GetBonds())])
        src = np.hstack([bonds[:,0], bonds[:,1]]) if len(bonds) > 0 else []
        dst = np.hstack([bonds[:,1], bonds[:,0]]) if len(bonds) > 0 else []
        molecular_graph = dgl.graph((src, dst), num_nodes = num_nodes)
        molecular_graph.ndata['attribute'] = node_attribute
        molecular_graph.ndata['type'] = node_type
        molecular_graph.edata['attribute'] = bond_attribute
        if self.use_3d:
            directions = node_position[dst] - node_position[src]
            molecular_graph.edata["length"] = directions.norm(dim=-1)
        graph_components.append(molecular_graph)
        assert 0 not in map_index, "Error: Some atoms in product can not be found in the reactants. It is expected that all the atom in product comes from reactant molecules."
        map_index = np.array(map_index) - 1
        product_index = np.arange(0,len(map_index)) + reactants.GetNumAtoms()
        edges = np.concatenate([map_index[:,np.newaxis],product_index[:,np.newaxis]],axis=-1)
        forward_edge_attribute = np.zeros([edges.shape[0],bond_attribute.size(-1)])
        backward_edge_attribute = np.zeros([edges.shape[0],bond_attribute.size(-1)])
        forward_edge_attribute[:,-3] = 1
        backward_edge_attribute[:,-2] = 1
        edge_attribute = np.concatenate([forward_edge_attribute,backward_edge_attribute])
        src = np.hstack([edges[:,0], edges[:,1]])
        dst = np.hstack([edges[:,1], edges[:,0]])
        reaction_edge_graph = dgl.graph((src, dst), num_nodes = num_nodes)
        reaction_edge_graph.edata['attribute'] = torch.tensor(edge_attribute)
        if self.use_3d:
            reaction_edge_graph.edata["length"] = torch.zeros([edges.shape[0]*2])
        graph_components.append(reaction_edge_graph)
        if self.use_3d:
            angular_edges = set()
            for i in range(molecular_graph.num_nodes()):
                predecessors = molecular_graph.predecessors(i)
                for j in predecessors:
                    for k in predecessors:
                        if j!=k:
                            angular_edges.add((j.item(),k.item()))
            src,dst = list(zip(*angular_edges)) if len(angular_edges) > 0 else ([],[])
            edge_attribute = np.zeros([len(angular_edges),bond_attribute.size(-1)])
            edge_attribute[:,-1] = 1
            src = np.array(src)
            dst = np.array(dst)
            angular_edge_graph = dgl.graph((src,dst),num_nodes=num_nodes)
            angular_edge_graph.edata['attribute'] = torch.tensor(edge_attribute)
            directions = node_position[dst] - node_position[src]
            angular_edge_graph.edata["length"] = directions.norm(dim=-1)
            graph_components.append(angular_edge_graph)

        reaction_graph = dgl.merge(graph_components)
        reaction_graph.gdata = {}
        reaction_graph.gdata["atom_per_molecule"] = atom_per_molecule
        reaction_graph.gdata["molecule_per_reaction"] = molecule_per_reaction
        return reaction_graph
    
    def __call__(self, smiles):
        if isinstance(smiles,str):
            return self.get_reaction_graph(smiles)
        elif isinstance(smiles,list):
            batch = [self.get_reaction_graph(item) for item in smiles]
            batch = utils.batch(batch)
            return batch