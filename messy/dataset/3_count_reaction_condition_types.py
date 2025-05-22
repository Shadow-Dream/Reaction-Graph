import os
import pandas as pd
import argparse
from multiprocessing import Pool
import json
def count_one_file(file_path):
    index = file_path[file_path.index("/") + 1:file_path.index(".")]
    index = int(index)
    dataset = pd.read_csv(file_path).fillna("")
    info = {"catalyst":{},"solvent":{},"agent":{},"atmosphere":{}}
    for _,data_row in dataset.iterrows():
        for info_key in info:
            for data_key in data_row.keys():
                if data_key.startswith(info_key):
                    data_value = data_row[data_key]
                    info[info_key][data_value] = info[info_key].get(data_value,0) + 1
    with open(f"stage3/{index}.json","w") as f:
        json.dump(info,f,indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument("--stage2_dir",type=str,default="stage2.1")
    args = parser.parse_args()

    stage2_dir = args.stage2_dir
    num_of_files = len(os.listdir(stage2_dir))
    pool = Pool(num_of_files)
    inputs = [f"{stage2_dir}/{i}.csv" for i in range(num_of_files)]
    results = pool.map(count_one_file,inputs)



