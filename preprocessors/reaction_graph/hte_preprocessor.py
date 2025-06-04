from encoders import ReactionGraphEncoder
import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import pickle as pkl
import os
import argparse
from metadatas import BUCHWALD_HARTWIG_METADATA, SUZUKI_MIYAURA_METADATA

class HTEPreprocessor:
    def __init__(self,
                 source_file="datasets/hte/buchwald_hartwig/split1/train.csv",
                 source_type="train",
                 target_dir="datasets/hte/buchwald_hartwig/split1",
                 target_name="ReactionGraphBuchwaldHartwigSplit1",
                 batch_size=32,
                 progress_bar = False,
                 log_delta = 100):
        self.progress_bar = progress_bar
        self.log_delta = log_delta
        self.source_type = source_type
        self.target_dir = target_dir
        self.target_name = target_name
        self.batch_size = batch_size
        self.dataset = pd.read_csv(source_file)
        self.dataset.fillna("",inplace=True)
        if "BuchwaldHartwig" in target_name:
            metadata = BUCHWALD_HARTWIG_METADATA
        else:
            metadata = SUZUKI_MIYAURA_METADATA
        self.encoder = ReactionGraphEncoder(metadata)

    def encode(self,row):
        reactant,product = row["mapped_smiles"].split(">>")
        reagent = row["others_smiles"]
        smiles = f"{reactant}>{reagent}>{product}"
        input = self.encoder(smiles)
        output = [row["yield"]]
        return input,output
    
    def process(self):
        progress = tqdm(self.dataset.iterrows(),desc=f"Processing {self.target_name}",total=len(self.dataset)) if self.progress_bar else self.dataset.iterrows()
        batches = []
        key_list = []
        input_list = []
        output_list = []
        error_list = []
        
        for index,row in progress:
            if not self.progress_bar and index % self.log_delta == 0:
                print(f"Processing {self.target_name} {self.source_type.capitalize()} {index}/{len(self.dataset)}")
            key = row["smiles"]
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
        
        target_name = f"{self.target_name}{self.source_type.capitalize()}.pkl"
        target_file = os.path.join(self.target_dir,target_name)
        with open(target_file,"wb") as f:
            pkl.dump(batches,f)
        target_error_name = f"{self.target_name}{self.source_type.capitalize()}Error.pkl"
        target_error_file = os.path.join(self.target_dir,target_error_name)
        with open(target_error_file,"wb") as f:
            pkl.dump(error_list,f)

        print(f"Processing {self.target_name} {self.source_type.capitalize()} Done!")

    def __call__(self):
        self.process()
    
parser = argparse.ArgumentParser(description='Reaction Graph HTE Preprocessor')
parser.add_argument('--source_file', type=str, default="datasets/hte/buchwald_hartwig/split1/train.csv")
parser.add_argument('--source_type', type=str, default="train")
parser.add_argument('--target_dir', type=str, default="datasets/hte/buchwald_hartwig/split1")
parser.add_argument('--target_name', type=str, default="ReactionGraphbuchwaldHartwigSplit1")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--progress_bar', type=bool, default=False)
parser.add_argument('--log_delta', type=int, default=100)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
preprocessor = HTEPreprocessor(args.source_file,args.source_type,args.target_dir,args.target_name,args.batch_size,args.progress_bar,args.log_delta)
preprocessor()