from tqdm import tqdm
from rdkit import Chem

def analyse_dataset(molecules):
    atomic_nums = set()
    degrees = set()
    formal_charges = set()
    chiral_tags = set()
    num_Hs = set()
    hybridizations = set()
    bond_types = set()
    stereos = set()
    for molecule in tqdm(molecules):
        molecule = Chem.MolFromSmiles(molecule)
        if molecule is None:
            continue
        for atom in molecule.GetAtoms():
            atomic_nums.add(atom.GetAtomicNum())
            degrees.add(atom.GetTotalDegree())
            formal_charges.add(atom.GetFormalCharge())
            chiral_tags.add(int(atom.GetChiralTag()))
            num_Hs.add(int(atom.GetTotalNumHs()))
            hybridizations.add(atom.GetHybridization())
        for bond in molecule.GetBonds():
            bond_types.add(bond.GetBondType())
            stereos.add(int(bond.GetStereo()))

    atomic_nums = list(atomic_nums)
    degrees = list(degrees)
    formal_charges = list(formal_charges)
    chiral_tags = list(chiral_tags)
    num_Hs = list(num_Hs)
    hybridizations = list(hybridizations)
    bond_types = list(bond_types)
    stereos = list(stereos)

    result = {
        "atomic_nums": atomic_nums,
        "degrees": degrees,
        "formal_charges": formal_charges,
        "chiral_tags": chiral_tags,
        "num_Hs": num_Hs,
        "hybridizations": hybridizations,
        "bond_types": bond_types,
        "stereos": stereos,
    }
    return result