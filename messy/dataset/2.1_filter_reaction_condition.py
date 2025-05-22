import pandas as pd
import argparse
import os
from multiprocessing import Pool

def process_one_file(inputs):
    file_path,config = inputs
    max_catalysts = config["max_catalysts"]
    max_solvents = config["max_solvents"]
    max_agents = config["max_agents"]
    max_atmospheres = config["max_atmospheres"]
    dataset = pd.read_csv(file_path).fillna("")
    preprocessed_dataset = []
    for data_index,data_row in dataset.iterrows():
        if data_index % 10000 == 0:
            with open(file_path.replace("stage1","stage2.1").replace(".csv",".log"),"w") as f:
                f.write(f"{data_index}/{len(dataset)}\n")

        catalysts = data_row["catalysts"]
        solvents = data_row["solvents"]
        agents = data_row["agents"]
        atmospheres = data_row["atmospheres"]

        catalysts = set(catalysts.split(","))
        solvents = set(solvents.split(","))
        agents = set(agents.split(","))
        atmospheres = set(atmospheres.split(","))

        if "" in catalysts:
            catalysts.remove("")
        if "" in solvents:
            solvents.remove("")
        if "" in agents:
            agents.remove("")
        if "" in atmospheres:
            atmospheres.remove("")

        if len(catalysts) > max_catalysts:
            continue
        elif len(solvents) > max_solvents:
            if len(catalysts)==0:
                continue
            solvents = ["invalid"] * max_solvents
            agents = ["invalid"] * max_agents
            atmospheres = ["invalid"] * max_atmospheres
        elif len(agents) > max_agents:
            if len(catalysts) + len(solvents)==0:
                continue
            agents = ["invalid"] * max_agents
            atmospheres = ["invalid"] * max_atmospheres
        elif len(atmospheres) > max_atmospheres:
            if len(catalysts) + len(solvents) + len(agents)==0:
                continue
            atmospheres = ["invalid"] * max_atmospheres
        else:
            if len(catalysts) + len(solvents) + len(agents) + len(atmospheres) == 0:
                continue

        catalysts = sorted(list(catalysts)) + [""]*(max_catalysts - len(catalysts))
        solvents = sorted(list(solvents)) + [""]*(max_solvents - len(solvents))
        agents = sorted(list(agents)) + [""]*(max_agents - len(agents))
        atmospheres = sorted(list(atmospheres)) + [""]*(max_atmospheres - len(atmospheres))

        preprocessed_row = {}
        preprocessed_row["smiles"] = data_row["smiles"]
        for i in range(max_catalysts):
            preprocessed_row[f"catalyst{i+1}"] = catalysts[i]
        for i in range(max_solvents):
            preprocessed_row[f"solvent{i+1}"] = solvents[i]
        for i in range(max_agents):
            preprocessed_row[f"agent{i+1}"] = agents[i]
        for i in range(max_atmospheres):
            preprocessed_row[f"atmosphere{i+1}"] = atmospheres[i]
        preprocessed_dataset.append(preprocessed_row)
    preprocessed_dataset = {key:[item[key] for item in preprocessed_dataset] for key in preprocessed_dataset[0]}
    preprocessed_dataset = pd.DataFrame(preprocessed_dataset)
    preprocessed_dataset.to_csv(file_path.replace("stage1","stage2.1"),index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument("--stage1_dir",type=str,default="stage1")
    parser.add_argument("--max_catalysts",type=int,default=1)
    parser.add_argument("--max_solvents",type=int,default=2)
    parser.add_argument("--max_agents",type=int,default=2)
    parser.add_argument("--max_atmospheres",type=int,default=0)
    args = parser.parse_args()

    config = {
        "max_catalysts":args.max_catalysts,
        "max_solvents":args.max_solvents,
        "max_agents":args.max_agents,
        "max_atmospheres":args.max_atmospheres
    }

    stage1_dir = args.stage1_dir
    num_of_datasets = len(os.listdir(stage1_dir))
    inputs = [(f"{stage1_dir}/{i}.csv",config) for i in range(num_of_datasets)]
    pool = Pool(num_of_datasets)
    pool.map(process_one_file,inputs)

