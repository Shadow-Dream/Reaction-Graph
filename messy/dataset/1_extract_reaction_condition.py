import os
from pistachio import io
from multiprocessing import Pool
import pandas as pd
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
                catalysts = set()
                solvents = set()
                agents = set()
                atmospheres = set()
                reaction_smiles = reaction.data["smiles"]
                if "|" in reaction_smiles:
                    reaction_smiles,_ = reaction_smiles.split(" ")
                reactants,extra_agents,products = reaction_smiles.split(">")
                if extra_agents!="":
                    extra_agents = canonicalize_smiles(extra_agents)
                    if extra_agents == "":
                        continue
                    agents.add(extra_agents)
                reactants = canonicalize_smiles(reactants)
                products = canonicalize_smiles(products)
                if reactants == "" or products == "":
                    continue
                reaction_smiles = f"{reactants}>>{products}"
                failed = False
                for component in reaction.components:
                    component_smiles = component.smiles
                    component_smiles = canonicalize_smiles(component_smiles)
                    if component_smiles == "":
                        failed = True
                        break
                    if component.role == "Catalyst":
                        catalysts.add(component_smiles)
                    elif component.role == "Solvent":
                        solvents.add(component_smiles)
                    elif component.role == "Agent":
                        agents.add(component_smiles)
                    elif component.role == "Atmosphere":
                        atmospheres.add(component_smiles)
                if failed:
                    continue
                catalysts = ",".join(catalysts)
                solvents = ",".join(solvents)
                agents = ",".join(agents)
                atmospheres = ",".join(atmospheres)
                reaction_dataset.append({"smiles":reaction_smiles, "catalysts":catalysts, "solvents":solvents, "agents":agents, "atmospheres":atmospheres})
            except:
                continue
    reaction_dataset = {key:[item[key] for item in reaction_dataset] for key in reaction_dataset[0]}
    reaction_dataset = pd.DataFrame(reaction_dataset)
    reaction_dataset.to_csv(f"condition/stage1/{index}.csv", index=False)
    print(f"Finish: {index}")

if __name__ == "__main__":
    print("All Started")
    database_dir = "data"
    num_of_process = 64

    file_paths = get_all_json_files(database_dir)
    batch_size = len(file_paths) // num_of_process
    file_path_batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]
    pool = Pool(num_of_process)
    pool.map(process_batch, enumerate(file_path_batches))
    print("All Finished")

