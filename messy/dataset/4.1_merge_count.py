import argparse
import os
import json
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser("params")
    parser.add_argument("--stage3_dir",type=str,default="stage3.1")
    args = parser.parse_args()
    stage3_dir = args.stage3_dir
    info_file_paths = [os.path.join(stage3_dir,file_name) for file_name in os.listdir(stage3_dir) if file_name.endswith(".json")]
    info = {"catalyst":{},"solvent":{},"agent":{},"atmosphere":{}}
    for info_file_path in tqdm(info_file_paths):
        with open(info_file_path,"r") as f:
            subset_info = json.load(f)
        for key in info:
            for sub_key in subset_info[key]:
                info[key][sub_key] = info[key].get(sub_key,0) + subset_info[key][sub_key]
    for key in info:
        info[key] = {k: v for k, v in sorted(info[key].items(), key=lambda item: item[1], reverse=True)}
    with open("stage4.1/info.json","w") as f:
        json.dump(info,f,indent=4)