import os
from pistachio import io
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem
#禁止rdkit所有输出
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return ''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return ''

def get_all_json_files(directory):
    json_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                json_files.append(file_path)
        
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            json_files.extend(get_all_json_files(dir_path))
    
    return json_files

def process_batch(inputs):
    index,file_paths = inputs
    print(f"Begin: {index}")
    reaction_dataset = []
    for file_path in file_paths:
        reactions = io.read_json(file_path)
        for reaction in reactions:
            try:
                reaction_smiles = reaction.data["smiles"]
                if "|" in reaction_smiles:
                    reaction_smiles,_ = reaction_smiles.split(" ")
                reactants,_,products = reaction_smiles.split(">")
                reactants = canonicalize_smiles(reactants)
                products = canonicalize_smiles(products)
                if reactants == "" or products == "":
                    continue
                reaction_type = reaction.data["namerxn"] if "namerxn" in reaction.data else "0"
                reaction_smiles = f"{reactants}>>{products}"
                reaction_dataset.append({"smiles":reaction_smiles, "type":reaction_type})
            except:
                continue
    reaction_dataset = {key:[item[key] for item in reaction_dataset] for key in reaction_dataset[0]}
    reaction_dataset = pd.DataFrame(reaction_dataset)
    reaction_dataset.to_csv(f"type/stage1/{index}.csv", index=False)
    print(f"Finish: {index}")

if __name__ == "__main__":
    print("All Started")
    database_dir = "data"
    num_of_process = 64

    file_paths = get_all_json_files(database_dir)
    batch_size = len(file_paths) // num_of_process
    file_path_batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]
    # process_batch((0, file_path_batches[0]))
    pool = Pool(num_of_process)
    pool.map(process_batch, enumerate(file_path_batches))
    print("All Finished")

