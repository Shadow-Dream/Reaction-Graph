import pandas as pd
import pickle as pkl
from cgr_featurizor import get_featurizor
from cgr_encoder import encode_reaction
from tqdm import tqdm
import dgl
import torch
import argparse
import re
import numpy as np

import pandas as pd
from lightning import pytorch as pl
from pathlib import Path
from tqdm import tqdm
from chemprop import data, featurizers, models, nn
import argparse
import os

# BATCH_SIZE = 32
# parser = argparse.ArgumentParser()
# parser.add_argument("--split_num",type=int,default=6)
# parser.add_argument("--split_index",type=int,default=0)
# parser.add_argument("--split_type",type=str,default="test")
# args= parser.parse_args()
# split_num = args.split_num
# split_index = args.split_index
# split_type = args.split_type

# with open("pistachio_metadata.pkl","rb") as f:
#     metadata = pkl.load(f)
# featurizor = get_featurizor(metadata)
# if split_type == "train":
#     dataset = pd.read_csv("../t5chem/reaction_type_dataset_train.csv")
# elif split_type == "val":
#     dataset = pd.read_csv("../t5chem/reaction_type_dataset_val.csv")
# else:
#     dataset = pd.read_csv("../t5chem/reaction_type_dataset_test.csv")
# dataset.fillna("",inplace=True)
# split_size = len(dataset) // split_num + 1
# dataset = dataset.iloc[split_index*split_size:(split_index+1)*split_size]
# dataset.reset_index(drop=True,inplace=True)

# smiles_list = dataset["mapped_smiles"].values
# labels = dataset[["category"]].values
# labels = np.array(labels).astype(int)

# train_data = [data.ReactionDatapoint.from_smi(smi, y) for smi, y in zip(tqdm(smiles_list), labels)]

# train_dset = data.ReactionDataset(train_data, featurizor)
# train_loader = data.build_dataloader(train_dset, num_workers=0,shuffle=False)
# batches = []
# for batch in tqdm(train_loader,total = len(train_loader)):
#     batches.append(batch)

# with open("dataset/CGRPistachioType" + split_type.capitalize() + str(split_index) + ".pkl","wb") as f:
#     pkl.dump(batches,f)

import pandas as pd
import pickle as pkl
from cgr_featurizor import get_featurizor
from cgr_encoder import encode_reaction
from tqdm import tqdm
import dgl
import torch
import argparse
import re
import numpy as np
import os
import pandas as pd
from lightning import pytorch as pl
from pathlib import Path
from tqdm import tqdm
from chemprop import data, featurizers, models, nn
import argparse
from copy import deepcopy
import random
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
with open("pistachio_metadata.pkl","rb") as f:
    metadata = pkl.load(f)
featurizor = get_featurizor(metadata)

batches_train = []
batches_val = []
batches_test = []
for i in tqdm(range(54)):
    if not os.path.exists(f"dataset/CGRPistachioTypeTrain{i}.pkl"):
        continue
    with open(f"dataset/CGRPistachioTypeTrain{i}.pkl","rb") as f:
        batches_train += pkl.load(f)
for i in tqdm(range(6)):
    if not os.path.exists(f"dataset/CGRPistachioTypeTest{i}.pkl"):
        continue
    with open(f"dataset/CGRPistachioTypeTest{i}.pkl","rb") as f:
        batches_test += pkl.load(f)
for i in tqdm(range(6)):
    if not os.path.exists(f"dataset/CGRPistachioTypeVal{i}.pkl"):
        continue
    with open(f"dataset/CGRPistachioTypeVal{i}.pkl","rb") as f:
        batches_val += pkl.load(f)

fdims = featurizor.shape # the dimensions of the featurizer, given as (atom_dims, bond_dims).
mp = nn.BondMessagePassing(*fdims,d_h=1024)
print(nn.agg.AggregationRegistry)
agg = nn.SumAggregation()

print(nn.PredictorRegistry)

ffn = nn.MulticlassClassificationFFN(input_dim=1024,hidden_dim=512,n_classes=12)

batch_norm = True

print(nn.metrics.MetricRegistry)

mpnn = models.MPNN(mp, agg, ffn, batch_norm, []).to("cuda:2")
loss_fn = CrossEntropyLoss()
import torch
from chemprop.schedulers import NoamLR
opt = torch.optim.Adam(mpnn.parameters(), 1e-4)

lr_sched = NoamLR(
    opt,
    2,
    200,
    len(batches_train),
    1e-4,
    1e-3,
    1e-4,
)
import pycm
for epoch in range(200):
    losses = []
    random.shuffle(batches_train)
    progress = tqdm(batches_train)
    mpnn.train()
    for index,batch in enumerate(progress):
        mpnn.optimizer_zero_grad(epoch,index,optimizer=opt)
       

        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg = deepcopy(bmg)
        bmg.to("cuda:2")
        V_d = V_d.to("cuda:2") if V_d is not None else None
        X_d = X_d.to("cuda:2") if X_d is not None else None
        targets = targets.long().to("cuda:2")
        # targets = embedd[targets]
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        weights = weights.to("cuda:2") if weights is not None else None
        lt_mask = lt_mask.to("cuda:2") if lt_mask is not None else None
        gt_mask = gt_mask.to("cuda:2") if gt_mask is not None else None

        Z = mpnn.fingerprint(bmg, V_d, X_d)
        preds = mpnn.predictor.train_step(Z)
        loss = mpnn.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        loss.backward()
        mpnn.optimizer_step(epoch,index,optimizer=opt)
        losses.append(loss.item())
        lr_sched.step()
        progress.set_postfix({"loss":sum(losses[-1000:])/len(losses[-1000:])})
    mpnn.eval()
    with torch.no_grad():
        fakes = []
        reals = []
        progress = tqdm(batches_val)
        for index,batch in enumerate(progress):
            targets = batch.Y
            bmg = deepcopy(batch.bmg)
            bmg.to("cuda:2")
            V_d = batch.V_d.to("cuda:2") if batch.V_d is not None else None
            X_d = batch.X_d.to("cuda:2") if batch.X_d is not None else None
            real = batch.Y
            pred = mpnn(bmg,V_d,X_d).argmax(-1).cpu()
            
            reals.append(real)
            fakes.append(pred)

        reals = torch.cat(reals).long()
        fakes = torch.cat(fakes)
        cm = pycm.ConfusionMatrix(actual_vector=reals.numpy().ravel(),predict_vector=fakes.numpy().ravel())
        print(cm.Overall_ACC,cm.Overall_CEN,cm.Overall_MCC)