import pickle as pkl
import torch
from torch import nn
from torch.nn import functional as func
from unimol_tools.models import UniMolModel
from dgl.nn.pytorch import GlobalAttentionPooling
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Unimol(nn.Module):
    def __init__(self,embedding_dim = 1024):
        super(Unimol,self).__init__()
        embedding_dim = embedding_dim//2
        self.gnn = UniMolModel(output_dim=embedding_dim, data_type="molecule", remove_hs=True)
        molecule_gate_nn = nn.Linear(embedding_dim,1)
        molecule_feat_nn = nn.Linear(embedding_dim,embedding_dim)
        self.readout = GlobalAttentionPooling(gate_nn = molecule_gate_nn, feat_nn=molecule_feat_nn)
    
    def forward(self,inputs):
        molecule_features = self.gnn(inputs['src_tokens'], inputs['src_distance'], inputs['src_coord'], inputs['src_edge_type'],return_repr=True,return_atomic_reprs=True)['atomic_reprs']
        molecule_features = torch.cat(molecule_features)
        reaction_features = self.readout(inputs["graphs"],molecule_features)
        reaction_features = reaction_features.reshape(reaction_features.shape[0]//2,-1)
        return reaction_features
    
class UnimolFeatureExtractor(nn.Module):
    def __init__(self, 
                 dim_catalyst = 54,
                 dim_solvent = 87,
                 dim_reagent = 235,
                 dim_hidden = 1024,
                 dim_message = 256):
        super(UnimolFeatureExtractor,self).__init__()
        self.reaction_nn = Unimol(dim_hidden)

        self.inputs = nn.ModuleList([
            nn.Sequential(nn.Linear(dim_catalyst,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_solvent,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_solvent,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_reagent,dim_message),nn.ReLU())])
        
        self.hiddens = nn.ModuleList([
            nn.Sequential(nn.Linear(dim_hidden + i * dim_message,dim_hidden),
                          nn.ReLU(),
                          nn.Linear(dim_hidden, dim_hidden),
                          nn.Tanh()) for i in range(5)])
        self.outputs = nn.ModuleList([
            nn.Linear(dim_hidden,dim_catalyst),
            nn.Linear(dim_hidden,dim_solvent),
            nn.Linear(dim_hidden,dim_solvent),
            nn.Linear(dim_hidden,dim_reagent),
            nn.Linear(dim_hidden,dim_reagent),
        ])
        self.dim_catalyst = dim_catalyst
        self.dim_solvent = dim_solvent
        self.dim_reagent = dim_reagent

    def embedding(self,inputs):
        features = self.reaction_nn(inputs)
        return features
    
    def catalyst1(self,features):
        return self.outputs[0](self.hiddens[0](features))
    
    def solvent1(self,features,catalyst1):
        catalyst1 = self.inputs[0](catalyst1.float())
        features = torch.cat([features,catalyst1],dim = -1)
        return self.outputs[1](self.hiddens[1](features))
    
    def solvent2(self,features,catalyst1,solvent1):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        features = torch.cat([features,catalyst1,solvent1],dim = -1)
        return self.outputs[2](self.hiddens[2](features))
    
    def reagent1(self,features,catalyst1,solvent1,solvent2):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        solvent2 = self.inputs[2](solvent2.float())
        features = torch.cat([features,catalyst1,solvent1,solvent2],dim = -1)
        return self.outputs[3](self.hiddens[3](features))
    
    def reagent2(self,features,catalyst1,solvent1,solvent2,reagent1):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        solvent2 = self.inputs[2](solvent2.float())
        reagent1 = self.inputs[3](reagent1.float())
        features = torch.cat([features,catalyst1,solvent1,solvent2,reagent1],dim = -1)
        return self.outputs[4](self.hiddens[4](features))
    
    @torch.no_grad()
    def forward(self,inputs,beams = [1,3,1,5,1]):
        features = self.embedding(inputs)
        batch_size = features.shape[0]

        catalyst1 = func.softmax(self.catalyst1(features),dim=-1)
        catalyst1_score,catalyst1_topk = torch.topk(catalyst1,beams[0],dim=-1)
        catalyst1_topk = func.one_hot(catalyst1_topk,self.dim_catalyst)

        solvent1_scores = []
        solvent1_topks = []
        for index in range(beams[0]):
            solvent1 = func.softmax(self.solvent1(features,catalyst1_topk[:,index]),dim=-1)
            solvent1_score,solvent1_topk = torch.topk(solvent1,beams[1],dim=-1)
            solvent1_topk = func.one_hot(solvent1_topk,self.dim_solvent)
            solvent1_scores.append(solvent1_score)
            solvent1_topks.append(solvent1_topk)
        solvent1_score = torch.concat(solvent1_scores,1)
        solvent1_topk = torch.concat(solvent1_topks,1)

        solvent2_scores = []
        solvent2_topks = []
        for index in range(beams[0]*beams[1]):
            solvent2 = func.softmax(self.solvent2(features,catalyst1_topk[:,index//beams[1]],solvent1_topk[:,index]),dim=-1)
            solvent2_score,solvent2_topk = torch.topk(solvent2,beams[2],dim=-1)
            solvent2_topk = func.one_hot(solvent2_topk,self.dim_solvent)
            solvent2_scores.append(solvent2_score)
            solvent2_topks.append(solvent2_topk)
        solvent2_score = torch.concat(solvent2_scores,1)
        solvent2_topk = torch.concat(solvent2_topks,1)

        reagent1_scores = []
        reagent1_topks = []
        for index in range(beams[0]*beams[1]*beams[2]):
            reagent1 = func.softmax(self.reagent1(features,catalyst1_topk[:,index//(beams[1]*beams[2])],solvent1_topk[:,index//beams[2]],solvent2_topk[:,index]),dim=-1)
            reagent1_score,reagent1_topk = torch.topk(reagent1,beams[3],dim=-1)
            reagent1_topk = func.one_hot(reagent1_topk,self.dim_reagent)
            reagent1_scores.append(reagent1_score)
            reagent1_topks.append(reagent1_topk)
        reagent1_score = torch.concat(reagent1_scores,1)
        reagent1_topk = torch.concat(reagent1_topks,1)

        reagent2_scores = []
        reagent2_topks = []
        for index in range(beams[0]*beams[1]*beams[2]*beams[3]):
            reagent2 = func.softmax(self.reagent2(features,catalyst1_topk[:,index//(beams[1]*beams[2]*beams[3])],solvent1_topk[:,index//(beams[2]*beams[3])],solvent2_topk[:,index//beams[3]],reagent1_topk[:,index]),dim=-1)
            reagent2_score,reagent2_topk = torch.topk(reagent2,beams[4],dim=-1)
            reagent2_topk = func.one_hot(reagent2_topk,self.dim_reagent)
            reagent2_scores.append(reagent2_score)
            reagent2_topks.append(reagent2_topk)
        reagent2_score = torch.concat(reagent2_scores,1)
        reagent2_topk = torch.concat(reagent2_topks,1)

        catalyst1_topk = catalyst1_topk.repeat(1,1,beams[1]*beams[2]*beams[3]*beams[4]).reshape(batch_size,-1,self.dim_catalyst)
        solvent1_topk = solvent1_topk.repeat(1,1,beams[2]*beams[3]*beams[4]).reshape(batch_size,-1,self.dim_solvent)
        solvent2_topk = solvent2_topk.repeat(1,1,beams[3]*beams[4]).reshape(batch_size,-1,self.dim_solvent)
        reagent1_topk = reagent1_topk.repeat(1,1,beams[4]).reshape(batch_size,-1,self.dim_reagent)

        catalyst1_score = catalyst1_score.unsqueeze(-1).repeat(1,1,beams[1]*beams[2]*beams[3]*beams[4]).reshape(batch_size,-1)
        solvent1_score = solvent1_score.unsqueeze(-1).repeat(1,1,beams[2]*beams[3]*beams[4]).reshape(batch_size,-1)
        solvent2_score = solvent2_score.unsqueeze(-1).repeat(1,1,beams[3]*beams[4]).reshape(batch_size,-1)
        reagent1_score = reagent1_score.unsqueeze(-1).repeat(1,1,beams[4]).reshape(batch_size,-1)

        _,catalyst1_topk = catalyst1_topk.max(-1)
        _,solvent1_topk = solvent1_topk.max(-1)
        _,solvent2_topk = solvent2_topk.max(-1)
        _,reagent1_topk = reagent1_topk.max(-1)
        _,reagent2_topk = reagent2_topk.max(-1)

        catalyst1_topk = catalyst1_topk.unsqueeze(-1)
        solvent1_topk = solvent1_topk.unsqueeze(-1)
        solvent2_topk = solvent2_topk.unsqueeze(-1)
        reagent1_topk = reagent1_topk.unsqueeze(-1)
        reagent2_topk = reagent2_topk.unsqueeze(-1)

        topk = torch.cat([catalyst1_topk,solvent1_topk,solvent2_topk,reagent1_topk,reagent2_topk],-1)
        score = catalyst1_score * solvent1_score * solvent2_score * reagent1_score * reagent2_score
        _,sort_indices= score.sort(-1,descending=True)
        batch_size,beam_sum = sort_indices.shape
        sort_indices = sort_indices + torch.arange(0,batch_size).to(sort_indices.device).unsqueeze(-1) * beam_sum
        sort_indices = sort_indices.view(-1)
        topk = topk.view(-1,5)[sort_indices].view(batch_size,-1,5)
        return topk

class UniMolReactionModel:
    def __init__(self,
                 dataset_train = None,
                 dataset_test = None,
                 dataset_val = None,
                 model_dir = "",
                 device = "cuda",
                 dim_catalyst: int = 54,
                 dim_solvent: int = 87,
                 dim_reagent: int = 235,
                 max_gradient = 1e2,
                 parameters_for_model = {},
                 parameters_for_optimizer = {"lr":0.0005, 
                                             "weight_decay":1e-10},
                 parameters_for_scheduler = {"mode":"min", 
                                             "factor":0.1, 
                                             "patience":5, 
                                             "min_lr":1e-7, 
                                             "verbose":True}):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val

        self.device = device
        self.model_dir = model_dir
        self.model = UnimolFeatureExtractor(dim_catalyst,dim_solvent,dim_reagent,**parameters_for_model)

        self.cross_entropy_loss = CrossEntropyLoss()

        self.optimizer = Adam(self.model.parameters(),**parameters_for_optimizer)
        self.scheduler = ReduceLROnPlateau(self.optimizer,**parameters_for_scheduler)

        self.split_lengths = [0,dim_catalyst,dim_solvent,dim_solvent,dim_reagent,dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]

        self.max_gradient = max_gradient

    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient)

    def load(self,filename):
        state_dict = torch.load(f"{self.model_dir}/{filename}")
        self.model.load_state_dict(state_dict)

    def save(self,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        torch.save(self.model.state_dict(),f"{self.model_dir}/{model_name}")
    
    def load_last(self):
        model_name = self.__class__.__name__.lower()
        model_name += "-last"
        model_name += ".ckpt"
        self.load(model_name)
    @torch.no_grad()
    def train(self,
              epoches = 100, 
              save_delta = 10,
              progress_bar = True,
              accumulation_steps = 4,
              smoothing = [0.9,0.8,0.8,0.7,0.7]
              ):
        for epoch in range(epoches):
            self.model.train()
            progress = tqdm(enumerate(self.dataset_train),total=len(self.dataset_train)) if progress_bar else self.dataset_train
            embeddings_list = []
            outputs_list = []
            for index,batch in progress:
                try:
                    inputs = batch["inputs"]
                    outputs = batch["outputs"]
                    features = self.model.embedding(inputs)
                    embeddings_list.append(features.detach().cpu())
                    outputs_list.append(outputs.detach().cpu())
                except:
                    pass
            with open("embeddings_train.pkl","wb") as f:
                pkl.dump(embeddings_list,f)
            with open("outputs_train.pkl","wb") as f:
                pkl.dump(outputs_list,f)
            break

class UnimolDataset:
    def __init__(self,dataset_dir,type,device = "cuda"):
        dataset_file_names = []
        for dataset_file_name in os.listdir(dataset_dir):
            if dataset_file_name.startswith(f"Unimol_{type.capitalize()}_") and not dataset_file_name.endswith("_Conformer.pkl"):
                dataset_file_names.append(dataset_file_name)
        dataset_file_names = sorted(dataset_file_names)
        self.dataset_buffer = []
        for dataset_file_name in tqdm(dataset_file_names):
            with open(f"{dataset_dir}/{dataset_file_name}","rb") as f:
                self.dataset_buffer += pkl.load(f)
        for index,batch in tqdm(enumerate(self.dataset_buffer)):
            new_batch = {}
            new_batch['src_tokens'] = torch.tensor(batch['src_tokens'])
            new_batch['src_distance'] = torch.tensor(batch['src_distance'])
            new_batch['src_coord'] = torch.tensor(batch['src_coord'])
            new_batch['src_edge_type'] = torch.tensor(batch['src_edge_type'])
            new_batch['graphs'] = batch['graphs']
            new_batch['outputs'] = torch.tensor(batch['outputs']).float()
            self.dataset_buffer[index] = new_batch
        self.dataset_offset = -1
        self.device = device

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def process(self,batch):
        inputs = {}
        inputs['src_tokens'] = batch['src_tokens'].to(self.device)
        inputs['src_distance'] = batch['src_distance'].to(self.device)
        inputs['src_coord'] = batch['src_coord'].to(self.device)
        inputs['src_edge_type'] = batch['src_edge_type'].to(self.device)
        inputs['graphs'] = batch['graphs'].to(self.device)
        outputs = batch['outputs'].to(self.device)
        return {"inputs":inputs,"outputs":outputs}
    
    def __len__(self):
        return len(self.dataset_buffer)

    def __iter__(self):
        return self

    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = 0
            self.shuffle_dataset()
            raise StopIteration
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch

dataset_train = UnimolDataset("/xxx/dataset","train")
dataset_test = UnimolDataset("/xxx/dataset","test")
dataset_val = UnimolDataset("/xxx/dataset","val")
model = UniMolReactionModel(dataset_train,dataset_test,dataset_val,"weights2")
model.load("unimolreactionmodel-49.ckpt")
model.model = model.model.to("cuda")
model.model_dir = "weights3"
model.train(save_delta=5)