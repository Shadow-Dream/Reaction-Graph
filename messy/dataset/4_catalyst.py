import pandas as pd
import json
catalyst_threshold = 500
dataset = pd.read_csv("condition/stage3/dataset.csv")
dataset.fillna("",inplace=True)
#统计catalyst1列的数据
catalyst_dict = {}
for catalyst in dataset["catalyst1"]:
    catalyst_dict[catalyst] = catalyst_dict.get(catalyst,0) + 1
#sort
catalyst_dict = {k:v for k,v in sorted(catalyst_dict.items(), key=lambda x:x[1], reverse=True)}
new_catalyst_dict = {key:item for key,item in catalyst_dict.items() if item > catalyst_threshold}
catalyst_dict = new_catalyst_dict
#删除catalyst1列中不在catalyst_dict中的数据
dataset = dataset[dataset["catalyst1"].isin(catalyst_dict)]
dataset.to_csv("condition/stage4/dataset.csv",index=False)
