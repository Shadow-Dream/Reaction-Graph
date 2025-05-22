import argparse
import os
import pandas as pd
import json
from multiprocessing import Pool

def process_one_file(inputs):
    file_path,info = inputs
    dataset = pd.read_csv(file_path).fillna("")
    #新增一列deleted
    dataset["deleted"] = False
    for data_index,data_row in dataset.iterrows():
        if data_index % 10000 == 0:
            with open(file_path.replace("stage2","stage5").replace(".csv",".log"),"w") as f:
                f.write(f"{data_index}/{len(dataset)}\n")
        valid = True
        for data_key in data_row.keys():
            for info_key in info:
                if data_key.startswith(info_key):
                    if data_row[data_key] not in info[info_key]:
                        valid = False
                    break
            if not valid:
                break
        if not valid:
            dataset.at[data_index,"deleted"] = True
    dataset = dataset[~dataset["deleted"]]
    dataset = dataset.drop(columns=["deleted"])
    dataset.to_csv(file_path.replace("stage2","stage5"),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument("--stage2_dir",type=str,default="stage2")
    parser.add_argument("--stage4_dir",type=str,default="stage4")
    parser.add_argument("--catalysts_threshold",type=int,default=1000)
    parser.add_argument("--solvents_threshold",type=int,default=10000)
    parser.add_argument("--agents_threshold",type=int,default=100000)
    parser.add_argument("--atmospheres_threshold",type=int,default=0)
    args = parser.parse_args()

    stage2_dir = args.stage2_dir
    stage4_dir = args.stage4_dir
    data_file_paths = [os.path.join(stage2_dir,file_name) for file_name in os.listdir(stage2_dir) if file_name.endswith(".csv")]
    info_file_path = os.path.join(stage4_dir,"info.json")

    with open(info_file_path,"r") as f:
        info = json.load(f)
    #删除小于指定频次的condition
    thresholds = {
        "catalyst":args.catalysts_threshold,
        "solvent":args.solvents_threshold,
        "agent":args.agents_threshold,
        "atmosphere":args.atmospheres_threshold
    }
    for key in info:
        info[key] = {k:v for k,v in info[key].items() if v >= thresholds[key]}
    inputs = [(file_path,info) for file_path in data_file_paths]
    pool = Pool(len(data_file_paths))
    pool.map(process_one_file,inputs)

