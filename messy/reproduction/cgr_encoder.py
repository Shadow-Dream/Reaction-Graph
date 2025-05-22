
from rdkit import Chem
import dgl
import torch
from rdkit.Chem import rdChemReactions

def test_center(smiles,reaction_featurizer):
    reaction = rdChemReactions.ReactionFromSmarts(smiles)
    reaction.Initialize()
    reactant_center_atoms = reaction.GetReactingAtoms()
    reactants = reaction.GetReactants()
    products = reaction.GetProducts()
    for center_atoms,reactant in zip(reactant_center_atoms,reactants):
        for atom_index in center_atoms:
            reactant_atom = reactant.GetAtomWithIdx(atom_index)
            if reactant_atom.HasProp("molAtomMapNumber")==0:
                continue
            mapping_index = reactant_atom.GetProp("molAtomMapNumber")
            for product in products:
                for product_atom in product.GetAtoms():
                    if product_atom.HasProp("molAtomMapNumber") == 0:
                        continue
                    if product_atom.GetProp("molAtomMapNumber") == mapping_index:
                        #对比reactant_atom和product_atom附近的原子是否相同
                        reactant_neighbors = [neighbor.GetSymbol() for neighbor in reactant_atom.GetNeighbors()]
                        product_neighbors = [neighbor.GetSymbol() for neighbor in product_atom.GetNeighbors()]
                        reactant_neighbors = sorted(reactant_neighbors)
                        product_neighbors = sorted(product_neighbors)
                        if reactant_neighbors == product_neighbors:
                            print(smiles)
                        

def encode_reaction_with_center(smiles,reaction_featurizer):
    try:
        reaction = rdChemReactions.ReactionFromSmarts(smiles,useSmiles=True)
        reaction.Initialize()
        reactant_center_atoms = reaction.GetReactingAtoms()
        reactants,products = smiles.split('>>')
        reactants = Chem.MolFromSmiles(reactants, sanitize=False)
        products = Chem.MolFromSmiles(products, sanitize=False)
        leaving_groups = []
        for atom in reactants.GetAtoms():
            leaving_groups.append(0 if atom.HasProp("molAtomMapNumber") else 1)
        for atom in products.GetAtoms():
            leaving_groups.append(0 if atom.HasProp("molAtomMapNumber") else 2)
        Chem.SanitizeMol(reactants, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        Chem.SanitizeMol(products, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        reactant_offsets = [0]
        for reactant_molecule in reaction.GetReactants()[:-1]:
            reactant_offsets.append(reactant_offsets[-1] + reactant_molecule.GetNumAtoms())

        reactant_center_atoms = [reactant_atom + reactant_offset
                                for reactant_offset,reactant_atoms 
                                in zip(reactant_offsets,reactant_center_atoms)
                                for reactant_atom 
                                in reactant_atoms]
        
        reaction = (reactants, products)
        
        cgr = reaction_featurizer(reaction)
        graph = dgl.graph((cgr.edge_index[0],cgr.edge_index[1]),num_nodes=cgr.V.shape[0])
        reaction_center = torch.zeros([cgr.V.shape[0],1],dtype=torch.uint8)
        reaction_center[reactant_center_atoms] = 1
        leaving_groups = torch.tensor(leaving_groups).view(-1,1)
        graph.ndata["attribute"] = torch.tensor(cgr.V)
        graph.ndata["center"] = reaction_center
        graph.ndata["leaving_group"] = leaving_groups
        graph.edata["attribute"] = torch.tensor(cgr.E)
        return graph
    except:
        return None

def encode_reaction(smiles,reaction_featurizer):
    try:
        reactants,products = smiles.split('>>')
        reactants = Chem.MolFromSmiles(reactants, sanitize=False)
        products = Chem.MolFromSmiles(products, sanitize=False)
        Chem.SanitizeMol(reactants, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        Chem.SanitizeMol(products, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        reaction = (reactants, products)
        
        cgr = reaction_featurizer(reaction)
        graph = dgl.graph((cgr.edge_index[0],cgr.edge_index[1]),num_nodes=cgr.V.shape[0])
        graph.ndata["attribute"] = torch.tensor(cgr.V)
        graph.edata["attribute"] = torch.tensor(cgr.E)
        return graph
    except:
        return None



