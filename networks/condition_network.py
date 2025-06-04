import torch
from torch import nn
from torch.nn import Module
from networks.components.rbf_mpnn import RBFMPNN
from torch.nn import functional as func
    
class ConditionNetwork(Module):
    def __init__(self, 
                 dim_catalyst = 54,
                 dim_solvent = 87,
                 dim_reagent = 235,
                 dim_hidden = 512,
                 dim_message = 256,
                 graph_type = "reaction_graph",
                 **parameters_for_reaction_nn):
        super(ConditionNetwork, self).__init__()

        if graph_type == "reaction_graph":
            self.reaction_nn = RBFMPNN(dim_hidden=dim_hidden,**parameters_for_reaction_nn)

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
    def inference(self,inputs,beams = [1,3,1,5,1]):
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