import pandas as pd
import pickle as pkl
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import openbabel

def get_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        mol_h = AllChem.AddHs(mol)
        try: 
            if AllChem.EmbedMolecule(mol_h, useRandomCoords = True, maxAttempts = 100) == -1:
                AllChem.Compute2DCoords(mol_h)
            else:
                if AllChem.MMFFOptimizeMolecule(mol_h) == -1:
                    AllChem.UFFOptimizeMolecule(mol_h)
            mol = AllChem.RemoveHs(mol_h)
            coordinates = np.array(mol.GetConformer().GetPositions())
        except:
            if AllChem.EmbedMolecule(mol_h, useRandomCoords = True, maxAttempts = 100) == -1:
                try:
                    obConversion = openbabel.OBConversion()
                    obConversion.SetInAndOutFormats("smi", "mol")
                    mol = openbabel.OBMol()
                    obConversion.ReadString(mol, smiles)
                    gen3D = openbabel.OBBuilder()
                    gen3D.Build(mol)
                    forcefield = openbabel.OBForceField.FindForceField("UFF")
                    forcefield.Setup(mol)
                    forcefield.ConjugateGradients(1000)
                    forcefield.GetCoordinates(mol)
                    num_atoms = mol.NumAtoms()
                    coordinates = []
                    for i in range(1, num_atoms + 1):
                        atom = mol.GetAtom(i)
                        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
                        coordinates.append([x, y, z])
                    coordinates = np.array(coordinates)
                except:
                    AllChem.Compute2DCoords(mol_h)
                    mol = AllChem.RemoveHs(mol_h)
                    coordinates = np.array(mol.GetConformer().GetPositions())
            else:
                mol = AllChem.RemoveHs(mol_h)
                coordinates = np.array(mol.GetConformer().GetPositions())
    except:
        coordinates = np.zeros((mol.GetNumAtoms(),3))
    return coordinates


parser = argparse.ArgumentParser()
parser.add_argument("--file",type = str,default="mapped_smiles/PistachioConditionTest0_9600.txt")
args = parser.parse_args()

with open(args.file,"r") as f:
    lines = f.read().splitlines()
coordinate_list = []
for reaction in tqdm(lines):
    reactants,products = reaction.split(">>")
    reactants = reactants.split(".")
    products = products.split(".")
    molecules = reactants + products
    coordinates = np.concatenate([get_conformer(molecule) for molecule in molecules],axis=0)
    coordinate_list.append(coordinates)
coordinate_list = np.concatenate(coordinate_list,axis=0)
np.save(args.file.replace("mapped_smiles","coordinates").replace(".txt",".npy"),coordinate_list)