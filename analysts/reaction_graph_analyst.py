from tqdm import tqdm
from rdkit import Chem

class ReactionGraphAnalyst:
    def __init__(self,progress_bar = False,log_delta = 1000,**kwargs):
        self.progress_bar = progress_bar
        self.log_delta = log_delta
        self.config = kwargs

    def analyze(self,reactions):
        progress = enumerate(tqdm(reactions,"Analyzing Dataset") if self.progress_bar else reactions)
        atom_list = set()
        charge_list = set()
        degree_list = set()
        hybridization_list = set()
        hydrogen_list = set()
        valence_list = set()
        bond_list = set()

        for index,reaction in progress:
            if not self.progress_bar and index % self.log_delta == 0:
                print(f"Analyzing Dataset {index}/{len(reactions)}")
            if "|" in reaction:
                reaction = reaction.split(" ")[0]
            mol = Chem.MolFromSmiles(reaction.replace(">>",".").replace(">","."))
            for atom in mol.GetAtoms():
                atom_list.add(atom.GetSymbol())
                charge_list.add(atom.GetFormalCharge())
                degree_list.add(atom.GetDegree())
                hybridization_list.add(str(atom.GetHybridization()))
                hydrogen_list.add(atom.GetTotalNumHs(includeNeighbors = True))
                valence_list.add(atom.GetTotalValence())
            
            for bond in mol.GetBonds():
                bond_list.add(str(bond.GetBondType()))

        atom_list = list(atom_list)
        charge_list = list(charge_list)
        degree_list = list(degree_list)
        hybridization_list = list(hybridization_list)
        hydrogen_list = list(hydrogen_list)
        valence_list = list(valence_list)
        bond_list = list(bond_list)
        
        atom_dim = len(atom_list)
        charge_dim = len(charge_list)
        degree_dim = len(degree_list)
        hybridization_dim = len(hybridization_list)
        hydrogen_dim = len(hydrogen_list)
        valence_dim = len(valence_list)
        bond_dim = len(bond_list)

        metadata = {
            "atom_list":atom_list,
            "charge_list":charge_list,
            "degree_list":degree_list,
            "hybridization_list":hybridization_list,
            "hydrogen_list":hydrogen_list,
            "valence_list":valence_list,
            "bond_list":bond_list,
            "atom_dim":atom_dim,
            "charge_dim":charge_dim,
            "degree_dim":degree_dim,
            "hybridization_dim":hybridization_dim,
            "hydrogen_dim":hydrogen_dim,
            "valence_dim":valence_dim,
            "bond_dim":bond_dim
        }

        metadata.update(self.config)
        return metadata

    def __call__(self,reactions):
        return self.analyze(reactions)