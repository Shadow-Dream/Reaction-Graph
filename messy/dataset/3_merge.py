import pandas as pd
import json
from tqdm import tqdm

dataset = pd.DataFrame()
for i in tqdm(range(64)):
    dataset = pd.concat([dataset,pd.read_csv(f"condition/stage2/{i}.csv")])
    
dataset.drop_duplicates(inplace=True)
dataset.to_csv("condition/stage3/dataset.csv",index=False)
