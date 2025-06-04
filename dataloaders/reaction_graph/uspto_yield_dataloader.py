import os
import numpy as np
import random
from tqdm import tqdm
import pickle as pkl
import torch

class USPTOYieldDataloader:
    def __init__(self, 
                 dataset_type = "",
                 dataset_dir = "",
                 dataset_name = "",
                 shuffle = False,
                 device = "cuda",
                 progress_bar = False,
                 load = True,
                 mean = 0,
                 std = 1,
                 var = 1,
                 **kwargs):
        
        self.dataset_dir = dataset_dir
        self.device = device
        self.shuffle = shuffle
        self.progress_bar = progress_bar
        self.mean = mean
        self.std = std
        self.var = var
        dataset_name += dataset_type

        dataset_filenames = os.listdir(dataset_dir)
        self.dataset_filenames = [dataset_filename 
            for dataset_filename in dataset_filenames 
            if dataset_filename.lower().startswith(dataset_name.lower()) 
            and dataset_filename.endswith(".pkl") 
            and "Error" not in dataset_filename]
        
        if load:
            self.load()

        if self.shuffle:
            self.shuffle_dataset()

    def load(self):
        self.dataset_index = 0
        self.dataset_offset = -1
        self.dataset_buffer = []

        progress = tqdm(self.dataset_filenames,desc = "Loading Datasets") if self.progress_bar else self.dataset_filenames
        for index,dataset_filename in enumerate(progress):
            dataset_filename = os.path.join(self.dataset_dir,dataset_filename)
            with open(dataset_filename,"rb") as f:
                self.dataset_buffer += pkl.load(f)
            if not self.progress_bar:
                print(f"Loaded {dataset_filename} split {index+1}/{len(self.dataset_filenames)}")

        self.length = len(self.dataset_buffer)
        
    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def process(self,batch):
        if isinstance(batch, dict):
            outputs = torch.tensor(batch["output"]).float().to(self.device)
            keys = batch["key"]
            reaction_graphs = self.process(batch["input"])
            processed_batch = {"input":reaction_graphs,"output":outputs,"key":keys}
            return processed_batch
        else:
            reaction_graphs = batch
            reaction_graphs = reaction_graphs.to(self.device)
            reaction_graphs.gdata = {
                k:v.to(self.device).long() for k,v in reaction_graphs.gdata.items()
            }
            attributes = reaction_graphs.ndata["attribute"].float()
            types = reaction_graphs.ndata["type"].float()
            type_mask = (types[:,0]>0) | (types[:,-1]>0)
            types[type_mask,0] = 1
            attributes = torch.cat([attributes,types[:,:-1]],dim = -1)
            reaction_graphs.ndata["attribute"] = attributes
            attribute = reaction_graphs.edata["attribute"].float()
            reaction_graphs.edata["attribute"] = attribute
            return reaction_graphs

            
    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = -1
            if self.shuffle:
                self.shuffle_dataset()
            raise StopIteration
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch
    
    def __call__(self, batch):
        return self.process(batch)