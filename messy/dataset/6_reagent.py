import pandas as pd
import json
from tqdm import tqdm
dataset = pd.read_csv("condition/stage5/dataset.csv")
#选出catalyst1列不是空的部分
subset = dataset[dataset["catalyst1"].notna()]
subset.fillna("",inplace=True)
#创建一个dict来统计agent1列的数据
agent_dict = {}
for agent in tqdm(subset["agent1"]):
    agent_dict[agent] = agent_dict.get(agent,0) + 1
#agent2列的数据
for agent in tqdm(subset["agent2"]):
    agent_dict[agent] = agent_dict.get(agent,0) + 1
#sort
agent_dict = {k:v for k,v in sorted(agent_dict.items(), key=lambda x:x[1], reverse=True)}
threshold = 100
new_agent_dict = {key:item for key,item in agent_dict.items() if item > threshold}
agent_dict = new_agent_dict
#删除agent1列和agent2列中不在agent_dict中的数据
dataset.fillna("",inplace=True)
dataset = dataset[dataset["agent1"].isin(agent_dict)]
dataset = dataset[dataset["agent2"].isin(agent_dict)]
dataset.to_csv("condition/stage6/dataset.csv",index=False)