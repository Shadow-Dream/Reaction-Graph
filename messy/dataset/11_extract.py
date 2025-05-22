import pandas as pd
from tqdm import tqdm
from rdkit import Chem

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return ''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''

dataset = pd.read_csv("condition/stage9/dataset.csv")
molecule_set = set()
for index,row in tqdm(dataset.iterrows(),total = len(dataset)):
    smiles = row["mapped_smiles"]
    reactants,products = smiles.split(">>")
    reactants = reactants.split(".")
    products = products.split(".")
    molecules = reactants + products
    molecules = [canonicalize_smiles(molecule) for molecule in molecules]
    molecule_set.update(molecules)
molecule_set = list(molecule_set)
with open("molecules.txt","w") as f:
    f.write("\n".join(molecule_set))

    