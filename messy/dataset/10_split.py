import pandas as pd
dataset = pd.read_csv("condition/stage9/dataset.csv")
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)
val = test.sample(frac=0.5)
test = test.drop(val.index)
train.to_csv("condition/stage10/train.csv",index=False)
test.to_csv("condition/stage10/test.csv",index=False)
val.to_csv("condition/stage10/val.csv",index=False)

