from encoders import ReactionGraphEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm
from . import utils
import copy
import pickle as pkl
import os
import argparse
from metadatas import USPTO_TPL_METADATA

class USPTOTPLPreprocessor:
    def __init__(self,
                 source_file="datasets/uspto_tpl/uspto_1k_TPL_train_valid.tsv.gzip",
                 source_type="train",
                 target_dir="datasets/uspto_tpl",
                 target_name="ReactionGraphUSPTOTPL",
                 split_num=16,
                 split_index=0,
                 batch_size=32,
                 progress_bar = False,
                 log_delta = 100,
                 split_random_seed = 42):
        self.split_random_seed = split_random_seed
        self.progress_bar = progress_bar
        self.log_delta = log_delta

        dataset = pd.read_csv(source_file, sep='\t', compression='gzip')
        dataset.fillna("",inplace=True)
        for col in ["Unnamed: 0", "level_0"]:
            if col in dataset.columns:
                dataset = dataset.drop(columns=[col])
        if source_type  in ["train","val"]:
            np.random.seed(self.split_random_seed)
            val_indices = []
            train_indices = []
            for label in dataset['labels'].unique():
                label_indices = dataset[dataset['labels'] == label].index.values
                n_val = max(1, int(0.1 * len(label_indices)))
                val_idx = np.random.choice(label_indices, size=n_val, replace=False)
                train_idx = np.setdiff1d(label_indices, val_idx)
                val_indices.extend(val_idx)
                train_indices.extend(train_idx)
            dataset.loc[val_indices, 'dataset'] = 'val'
            dataset.loc[train_indices, 'dataset'] = 'train'
            dataset = dataset[dataset["dataset"]==source_type]
        self.source_type = source_type
        self.target_dir = target_dir
        self.target_name = target_name
        self.split_index = split_index
        self.batch_size = batch_size
        split_size = len(dataset)//split_num + 1
        self.dataset = dataset[split_size*split_index:split_size*(split_index + 1)].reset_index(drop=True)
        self.num_types = np.array(USPTO_TPL_METADATA["num_types"])
        self.encoder = ReactionGraphEncoder(USPTO_TPL_METADATA)

    def encode(self,row):
        input = self.encoder(row["mapped_rxn"])
        output = np.zeros(self.num_types)
        output[row["labels"]] = 1
        return input,output
    
    def process(self):
        progress = tqdm(self.dataset.iterrows(),total = len(self.dataset),desc=f"Processing {self.target_name} Split {self.split_index}") if self.progress_bar else self.dataset.iterrows()
        batches = []
        key_list = []
        input_list = []
        output_list = []
        error_list = []
        
        for index,row in progress:
            if not self.progress_bar and index % self.log_delta == 0:
                print(f"Processing {self.target_name} {self.source_type.capitalize()} Split {self.split_index} {index}/{len(self.dataset)}")
                
            key = row["canonical_rxn"]
            try:
                input,output = self.encode(row)
                key_list.append(key)
                input_list.append(input)
                output_list.append(output)
            except Exception as e:
                error_list.append((key,e))

            if len(key_list) == self.batch_size:
                keys = copy.deepcopy(key_list)
                inputs = utils.batch(input_list)
                outputs = np.array(output_list)
                batch = {"key":keys,"input":inputs,"output":outputs}
                batches.append(batch)
                key_list = []
                input_list = []
                output_list = []

        if len(key_list) > 0:
            keys = copy.deepcopy(key_list)
            inputs = utils.batch(input_list)
            outputs = np.array(output_list)
            batch = {"key":keys,"input":inputs,"output":outputs}
            batches.append(batch)
        
        target_name = f"{self.target_name}{self.source_type.capitalize()}{self.split_index}.pkl"
        target_file = os.path.join(self.target_dir,target_name)
        with open(target_file,"wb") as f:
            pkl.dump(batches,f)
        target_error_name = f"{self.target_name}{self.source_type.capitalize()}Error{self.split_index}.pkl"
        target_error_file = os.path.join(self.target_dir,target_error_name)
        with open(target_error_file,"wb") as f:
            pkl.dump(error_list,f)

        print(f"Processing {self.target_name} {self.source_type.capitalize()} Split {self.split_index} Done!")

    def __call__(self):
        self.process()
    
parser = argparse.ArgumentParser(description='Reaction Graph USPTO TPL Preprocessor')
parser.add_argument('--source_file', type=str, default="datasets/uspto_tpl/uspto_1k_TPL_train_valid.tsv.gzip")
parser.add_argument('--source_type', type=str, default="train")
parser.add_argument('--target_dir', type=str, default="datasets/uspto_tpl")
parser.add_argument('--target_name', type=str, default="ReactionGraphUSPTOTPL")
parser.add_argument('--split_num', type=int, default=16)
parser.add_argument('--split_index', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--progress_bar', type=bool, default=False)
parser.add_argument('--log_delta', type=int, default=100)
parser.add_argument('--split_random_seed', type=int, default=42)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
preprocessor = USPTOTPLPreprocessor(args.source_file,args.source_type,args.target_dir,args.target_name,args.split_num,args.split_index,args.batch_size,args.progress_bar,args.log_delta,args.split_random_seed)
preprocessor()