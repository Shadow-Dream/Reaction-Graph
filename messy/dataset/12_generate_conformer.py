import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import openbabel
import sys
import contextlib
import io
import pickle as pkl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--split_num", type=int, default=64)
parser.add_argument("--split_index", type=int, default=0)
args = parser.parse_args()
split_num = args.split_num
split_index = args.split_index
with open("molecules.txt","r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    split_size = len(lines) // split_num + 1
    lines = lines[split_index*split_size:(split_index+1)*split_size]
conformer_keys = []
conformer_slices = [0]
conformer_datas = []

for line in tqdm(lines):
    smiles = line
    try:
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
        if coordinates is None:
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
        continue
    conformer_keys.append(line)
    conformer_slices.append(conformer_slices[-1] + len(coordinates))
    conformer_datas.append(coordinates)

conformer_keys = {key:i for i,key in enumerate(conformer_keys)}
conformer_slices = np.array(conformer_slices)
conformer_datas = np.concatenate(conformer_datas, axis=0)

with open(f"final/conformer/keys_{split_index}.pkl", "wb") as f:
    pkl.dump(conformer_keys, f)
np.save(f"final/conformer/slices_{split_index}.npy", conformer_slices)
np.save(f"final/conformer/datas_{split_index}.npy", conformer_datas)
