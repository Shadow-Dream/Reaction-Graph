import argparse
import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import pickle as pkl


def process_one_file(inputs):
    file_path,info = inputs
    dataset = pd.read_csv(file_path).fillna("")
    #新增一列deleted
    dataset["deleted"] = False
    for data_index,data_row in dataset.iterrows():
        if data_index % 10000 == 0:
            with open(file_path.replace("stage2.1","stage3.2").replace(".csv",".log"),"w") as f:
                f.write(f"{data_index}/{len(dataset)}\n")
        valid = True
        for data_key in data_row.keys():
            for info_key in info:
                if data_key.startswith(info_key):
                    if data_row[data_key]!="" and data_row[data_key] not in info[info_key]:
                        valid = False
                    break
            if not valid:
                break
        if not valid:
            dataset.at[data_index,"deleted"] = True
    dataset = dataset[~dataset["deleted"]]
    dataset = dataset.drop(columns=["deleted"])
    dataset.to_csv(file_path.replace("stage2.1","stage3.2"),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument("--pistachio_dir",type=str,default="stage2.1")
    parser.add_argument("--uspto_dir",type=str,default="../dataset")
    args = parser.parse_args()

    pistachio_dir = args.pistachio_dir
    uspto_dir = args.uspto_dir

    with open(f"{uspto_dir}/USPTO_Condition_Solvent.pkl","rb") as f:
        solvents = np.array(pkl.load(f))
    with open(f"{uspto_dir}/USPTO_Condition_Catalyst.pkl","rb") as f:
        catalysts = np.array(pkl.load(f))
    with open(f"{uspto_dir}/USPTO_Condition_Reagent.pkl","rb") as f:
        reagents = np.array(pkl.load(f))
    
    catalysts = np.delete(catalysts,np.where(catalysts == "nan"))
    solvents = np.delete(solvents,np.where(solvents == "nan"))
    reagents = np.delete(reagents,np.where(reagents == "nan"))
    info = {
        "catalyst":catalysts,
        "solvent":solvents,
        "agent":reagents
    }
    data_file_paths = [os.path.join(pistachio_dir,file_name) for file_name in os.listdir(pistachio_dir) if file_name.endswith(".csv")]
    
    inputs = [(file_path,info) for file_path in data_file_paths]
    pool = Pool(len(data_file_paths))
    pool.map(process_one_file,inputs)

