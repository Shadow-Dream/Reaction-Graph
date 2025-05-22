from tqdm import tqdm
import pandas as pd

dataset = pd.read_csv("condition/stage9/dataset.csv")
dataset.fillna("",inplace = True)
catalyst_dict = set()
solvent_dict = set()
agent_dict = set()

for index,row in tqdm(dataset.iterrows(),total = len(dataset)):
    catalyst = row["catalyst1"]
    catalyst_dict.add(catalyst)
    solvent = row["solvent1"]
    solvent_dict.add(solvent)
    solvent = row["solvent2"]
    solvent_dict.add(solvent)
    agent = row["agent1"]
    agent_dict.add(agent)
    agent = row["agent2"]
    agent_dict.add(agent)

catalyst_dict = list(catalyst_dict)
solvent_dict = list(solvent_dict)
agent_dict = list(agent_dict)
with open("catalyst.txt","w") as f:
    f.write("\n".join(catalyst_dict))
with open("solvent.txt","w") as f:
    f.write("\n".join(solvent_dict))
with open("agent.txt","w") as f:
    f.write("\n".join(agent_dict))