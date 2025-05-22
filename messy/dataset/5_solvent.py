import pandas as pd
import json
from tqdm import tqdm
dataset = pd.read_csv("condition/stage4/dataset.csv")
#选出catalyst1列不是空的部分
subset = dataset[dataset["catalyst1"].notna()]
subset.fillna("",inplace=True)
#创建一个dict来统计solvent1列的数据
solvent_dict = {}
for solvent in tqdm(subset["solvent1"]):
    if solvent=="":
        continue
    solvent_dict[solvent] = solvent_dict.get(solvent,0) + 1
#solvent2列的数据
for solvent in tqdm(subset["solvent2"]):
    if solvent=="":
        continue
    solvent_dict[solvent] = solvent_dict.get(solvent,0) + 1
#sort
solvent_dict = {k:v for k,v in sorted(solvent_dict.items(), key=lambda x:x[1], reverse=True)}
threshold = 100
new_solvent_dict = {key:item for key,item in solvent_dict.items() if item > threshold}
solvent_dict = new_solvent_dict
#删除solvent1列和solvent2列中不在solvent_dict中的数据
dataset.fillna("",inplace=True)
solvent_dict[""] = 0
dataset = dataset[dataset["solvent1"].isin(solvent_dict)]
dataset = dataset[dataset["solvent2"].isin(solvent_dict)]
dataset.to_csv("condition/stage5/dataset.csv",index=False)