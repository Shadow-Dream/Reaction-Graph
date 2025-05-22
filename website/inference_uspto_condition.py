import os
import random
import warnings

from argparse import ArgumentParser
import numpy as np

import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as func

import dgl
from dgl.nn.pytorch import NNConv

from rxnmapper import RXNMapper
from itertools import groupby

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures,AllChem,rdChemReactions
from dgl.readout import sum_nodes, broadcast_nodes,softmax_nodes
import json

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")


attention_weight = None

class Set2Set(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        global attention_weight
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                if attention_weight is None:
                    attention_weight = alpha.cpu().numpy()
                
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = torch.cat([q, readout], dim=-1)

            return q_star

    def extra_repr(self):
        summary = 'n_iters={n_iters}'
        return summary.format(**self.__dict__)

class RBFLayer(torch.nn.Module):
    def __init__(self, dim):
        super(RBFLayer, self).__init__()
        self.dim = dim
        self.centers = torch.nn.Parameter(torch.Tensor(dim))
        self.beta = torch.nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.centers, 0, 1)
        torch.nn.init.constant_(self.beta, 10)

    def forward(self, x):
        x = x.view(-1, 1)
        centers = self.centers.view(1, -1)
        diff = x - centers
        dist_sq = torch.square(diff)
        out = torch.exp(-self.beta * dist_sq)
        return out

class ReactionNN(Module):
    def __init__(self,
                 dim_node_attribute = 110,
                 dim_edge_attribute = 8,
                 dim_edge_length = 8,
                 dim_hidden_features = 64,
                 dim_hidden = 4096,
                 message_passing_step = 4,
                 pooling_step = 3,
                 num_layers_pooling = 2):
        
        super(ReactionNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(
            nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
        self.rbf = RBFLayer(dim_edge_length)
        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
    
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)

        self.pooling = Set2Set(input_dim = dim_hidden_features * 2,
                               n_iters = pooling_step,
                               n_layers = num_layers_pooling)

        self.sparsify = nn.Sequential(
            nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU()
        )

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step
    
    def get_node_features(self,inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]

        node_attribute = message_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        
        edge_attribute = message_passing_graph.edata["attribute"]
        edge_length = self.rbf(message_passing_graph.edata["length"])
        edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(message_passing_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        return node_aggregation
    
    def get_pooling(self,inputs,node_features):
        message_passing_graph = inputs["message_passing_graph"]
        reaction_features = self.pooling(message_passing_graph, node_features)
        reaction_features = self.sparsify(reaction_features)
        return reaction_features

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]

        node_attribute = message_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        
        edge_attribute = message_passing_graph.edata["attribute"]
        edge_length = self.rbf(message_passing_graph.edata["length"])
        edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(message_passing_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        reaction_features = self.pooling(message_passing_graph, node_aggregation)
        reaction_features = self.sparsify(reaction_features)
        return reaction_features
    
class GraphFeatureExtractor(Module):
    def __init__(self, 
                 dim_catalyst = 54,
                 dim_solvent = 87,
                 dim_reagent = 235,
                 dim_hidden = 512,
                 dim_message = 256,
                 last_bias = True,
                 **parameters_for_reaction_nn):
        super(GraphFeatureExtractor, self).__init__()
        
        self.reaction_nn = ReactionNN(dim_hidden=dim_hidden,**parameters_for_reaction_nn)
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
            nn.Linear(dim_hidden,dim_catalyst,bias=last_bias),
            nn.Linear(dim_hidden,dim_solvent,bias=last_bias),
            nn.Linear(dim_hidden,dim_solvent,bias=last_bias),
            nn.Linear(dim_hidden,dim_reagent,bias=last_bias),
            nn.Linear(dim_hidden,dim_reagent,bias=last_bias),
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
    def forward(self,fingerprint,beams = [1,3,1,5,1]):
        features = self.embedding(fingerprint)
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

class ReactionGraphModel:
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
                 parameters_for_optimizer_pretrain = {"lr":0.0005, 
                                             "weight_decay":1e-10},
                 parameters_for_scheduler_pretrain = {"mode":"min", 
                                             "factor":0.1, 
                                             "patience":20, 
                                             "min_lr":1e-7, 
                                             "verbose":True},
                 parameters_for_optimizer_finetune = {"lr":0.0005, 
                                             "weight_decay":1e-10},
                 parameters_for_scheduler_finetune = {"mode":"min", 
                                             "factor":0.1, 
                                             "patience":20, 
                                             "min_lr":1e-7, 
                                             "verbose":True},
                 none_weights = {"catalyst1":0.1,
                                "solvent1":1,
                                "solvent2":0.1,
                                "reagent1":1,
                                "reagent2":0.1},
                 smoothing = [0.9,0.8,0.8,0.7,0.7],
                 accumulation_steps = 4):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.max_gradient = max_gradient
        self.device = device
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.smoothing = smoothing
        self.accmulation_steps = accumulation_steps
        
        self.model = GraphFeatureExtractor(dim_catalyst,dim_solvent,dim_reagent,**parameters_for_model).to(device)

        self.split_lengths = [0,dim_catalyst,dim_solvent,dim_solvent,dim_reagent,dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]

    def load(self,filename):
        state_dict = torch.load(f"{self.model_dir}/{filename}")
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def inference(self,inputs):
        return self.model(inputs)

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
atom_list = ['Ag','Al','As','B','Bi',
             'Br','C','Cl','Co','Cr',
             'Cu','F','Ge','H','I',
             'In','K','Li','Mg','Mo',
             'N','Na','O','P','Pd',
             'S','Sb','Se','Si','Sn',
             'Te','Zn','Os','Ti','Xe',
             'Ga','Ca','Zr','Gd','Rh',
             'Rb','Ba','Ce','La','Tb', 
             'V','Mn','Sm','W','Ru', 
             'Pt','Tl','Ni','Pb','Cd', 
             'Y','U','Hf','Fe','Hg','Au']

charge_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
degree_list = [1, 2, 3, 4, 5, 6, 7, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


catalysts = ['Cc1ccc(S(=O)(=O)O)cc1', 'CC(C)(C#N)N=NC(C)(C)C#N', 'c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1', 'O=[Mn]=O', 'Cl[Fe](Cl)Cl', 'I[Cu]I', 'CN(C)C=O', 'O=C(O)C(F)(F)F', '[Cu]Br', '[Pd]', '[Zn]', 'Cc1ccccc1[P](c1ccccc1C)(c1ccccc1C)[Pd](Cl)(Cl)[P](c1ccccc1C)(c1ccccc1C)c1ccccc1C', 'CC(=O)O[Pd]OC(C)=O', 'CN(C)c1ccccn1', 'II', 'Cl[Pd]Cl', 'C1COCCO1', '[Ru]', '[Cu]', 'O=[Os](=O)(=O)=O', "", 'C1COCCOCCOCCOCCOCCO1', 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1', 'CC(=O)O', 'Cl[Ti](Cl)(Cl)Cl', 'O=[Cu-]', 'Cl', 'CN(C)c1ccncc1', 'CCN(CC)CC', 'Cl[Cu]', 'CO', 'CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C', 'Cl[Hg]Cl', '[Pt]', '[Fe]', 'Cl[Ni]Cl', 'Cc1cc(C)c(N2CCN(c3c(C)cc(C)cc3C)C2=[Ru](Cl)(Cl)(=Cc2ccccc2)[P](C2CCCCC2)(C2CCCCC2)C2CCCCC2)c(C)c1', 'O=[Ag-]', 'O=[Pt]=O', '[Cu]I', 'CC(C)O[Ti](OC(C)C)(OC(C)C)OC(C)C', 'Cl[Cu]Cl', 'O=[Ag]', 'C1CCNCC1', 'Br[Cu]Br', 'O=[Pt]', 'c1ccncc1', 'O', 'O=S(=O)(O)O', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', 'O=[Cu]', 'O=C(OOC(=O)c1ccccc1)c1ccccc1', '[Rh]', '[Ni]']

solvents = ['CCC#N', 'ClC(Cl)Cl', '[Cl-]', 'OCC(F)(F)F', 'CCCCO', 'S=C=S', 'COC(C)(C)C', 'N', 'Cc1ccc(C)cc1', 'CN1CCCN(C)C1=O', '[NH4+]', 'c1ccc(Oc2ccccc2)cc1', 'ClCCl', '[O-][Cl+3]([O-])([O-])O', 'Br', 'C1CCCCC1', 'Clc1ccccc1', 'CC1CCCO1', 'CCOC(C)=O', '[Na+]', 'COc1ccccc1', 'CCCCCO', 'O=S(Cl)Cl', 'CCC(C)O', 'O=[N+]([O-])c1ccccc1', 'O=C([O-])O', 'Clc1ccccc1Cl', 'CCO', '[K+]', 'O=S1(=O)CCCC1', 'O=C(O)C(F)(F)F', 'O=P([O-])([O-])[O-]', 'c1ccc2ncccc2c1', 'Cc1ccccc1C', 'COCCOCCOC', 'CCC(C)=O', 'C1COCCO1', '[OH-]', 'O=CO', 'O=P(Cl)(Cl)Cl', 'CC(C)OC(C)C', 'CCCO', 'c1ccncc1', 'O', 'COCCO', 'C1CCOC1', 'ClCCCl', 'c1ccccc1', 'C[N+](=O)[O-]', 'CCCCCC', 'Cc1ccccc1', 'CCOCCO', 'CN(C)C=O', 'OCCO', 'CC(Cl)Cl', 'CCOCC', 'CC(C)CO', 'CCN(C(C)C)C(C)C', 'CCC(C)(C)O', 'CO', 'CCCCC', 'OCCOCCO', 'CC#N', 'Cc1cc(C)cc(C)c1', 'C1CCNCC1', 'CC(=O)CC(C)C', 'CN(C)P(=O)(N(C)C)N(C)C', 'COCCOC', 'CC(=O)OC(C)=O', 'CCCCCCC', 'CC(=O)OC(C)C', 'CC(C)(C)O', 'CN1CCCC1=O', 'CC(C)O', 'CC(C)CCO', 'CS(C)=O', 'OCC(O)CO', "", 'CC(=O)O', 'CS(=O)(=O)O', 'Cl', 'CCN(CC)CC', 'CC(C)=O', 'CCNCC', 'CC(=O)N(C)C', 'ClC(Cl)(Cl)Cl', 'O=S(=O)(O)O']

reagents = ['CN(C)C(On1nnc2ccccc21)=[N+](C)C.F[P-](F)(F)(F)(F)F', 'O=C(N=NC(=O)N1CCCCC1)N1CCCCC1', '[Li]O', 'CC(C)COC(=O)Cl', '[Na+].[OH-]', 'ClC(Cl)Cl', 'CC(C)C[AlH]CC(C)C', 'CC(C)(C)ON=O', '[Li]C(C)(C)C', '[BH4-].[Li+]', 'Br[P+](N1CCCC1)(N1CCCC1)N1CCCC1.F[P-](F)(F)(F)(F)F', 'CCCCO', '[Mg]', '[Br-].[K+]', 'COC(C)(C)C', 'CC(Cl)OC(=O)Cl', '[Al+3].[Cl-].[Cl-].[Cl-]', 'CCCCN(CCCC)CCCC', 'N', 'O=C[O-].[NH4+]', 'C[Si](C)(C)C=[N+]=[N-]', 'N[C@@H]1CCCC[C@H]1N', 'c1ccc(Oc2ccccc2)cc1', 'C1=CCCCC1', 'ClCCl', 'ClP(Cl)(Cl)(Cl)Cl', 'CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C.F[P-](F)(F)(F)(F)F', '[I-].[Na+]', 'O=C(O)O', '[O-][Cl+3]([O-])([O-])O', '[H][H]', 'Br', 'c1ccc(P(c2ccccc2)c2ccccc2)cc1', 'O=C(n1ccnc1)n1ccnc1', 'CCCC[N+](CCCC)(CCCC)CCCC.[F-]', 'O=C([O-])[O-].[K+].[K+]', '[Cl-].[NH4+]', '[BH4-].[Na+]', 'O=C(CO)O[CH][C@@H](O)CO', 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1', 'CS(=O)(=O)Cl', 'C1CCC2=NCCCN2CC1', 'O=C([O-])O.[K+]', 'c1c[nH]cn1', 'CCOC(C)=O', '[I-].[Li+]', 'O=C1CCC(=O)N1Br', 'c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1', 'O=C=O', 'COc1ccccc1', 'O=S(Cl)Cl', 'O=C(O)C(=O)O', 'CCCCP(CCCC)CCCC', 'C[Si](C)(C)[N-][Si](C)(C)C.[Li+]', 'On1nnc2ccccc21', 'CC(=O)[O-].[Na+]', 'CC1(C)CCCC(C)(C)N1', 'O=C([O-])[O-].[Li+]', 'Cc1ccc(S(=O)(=O)O)cc1', 'CCCCP(=CC#N)(CCCC)CCCC', 'CC(C)(C#N)N=NC(C)(C)C#N', 'I', 'ClP(Cl)Cl', 'CC(C)(C)[O-].[K+]', 'ClB(Cl)Cl', 'CCO', '[K+].[OH-]', 'CN(C)C(N(C)C)=[N+]1N=[N+]([O-])c2ncccc21.F[P-](F)(F)(F)(F)F', 'CCOC(=O)/N=N/C(=O)OCC', 'C', 'O=C(c1ncc[nH]1)c1ncc[nH]1', 'Cc1ccccc1S(=O)(=O)O', 'CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21', 'O=C(O)C(F)(F)F', '[I-].[K+]', 'C1COCCOCCOCCOCCO1', '[Al+3].[H-].[H-].[H-].[H-].[Li+]', 'N#N', 'c1ccc2ncccc2c1', 'CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]', '[BH3-]C#N.[Na+]', '[Li]', '[C-]#N.[Na+]', 'Cc1ccccc1C', 'C1COCCO1', 'O=CO', 'CN', 'O=C(O)CC(O)(CC(=O)O)C(=O)O', 'O=P([O-])([O-])[O-].[K+].[K+].[K+]', 'NN', 'O=S(=O)(O)C(F)(F)F', 'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1', 'C1COCCOCCOCCOCCOCCO1', 'CC(C)CCON=O', 'CC(C)OC(=O)/N=N/C(=O)OC(C)C', 'OO', 'O=P12OP3(=O)OP(=O)(O1)OP(=O)(O2)O3', 'O=S([O-])S(=O)[O-].[Na+].[Na+]', 'CCCP1(=O)OP(=O)(CCC)OP(=O)(CCC)O1', 'CN(C)CCN(C)C', 'O=P(Cl)(Cl)Cl', 'CCOP(=O)(C#N)OCC', 'O=S(=O)([O-])C(F)(F)F.O=S(=O)([O-])C(F)(F)F.O=S(=O)([O-])C(F)(F)F.[Yb+3]', 'N#CC1=C(C#N)C(=O)C(Cl)=C(Cl)C1=O', 'C[Al](C)C', 'CC(C)OC(C)C', 'C1CN2CCN1CC2', 'On1nnc2cccnc21', 'O=N[O-].[Na+]', 'O=S(=O)([O-])[O-].[Mg+2]', '[Na+].[O-]Cl', 'CC(C)[Mg]Cl', '[NH4+].[OH-]', 'c1ccncc1', 'O', 'CN(C)C(On1nnc2ccccc21)=[N+](C)C.F[B-](F)(F)F', 'C1CCOC1', 'c1cnc2c(c1)ccc1cccnc12', 'CC[Mg]Br', 'ClCCCl', 'c1ccccc1', 'CC(=O)Cl', 'O=C(OO)c1cccc(Cl)c1', 'CC=C(C)C', '[Na+].[O-][I+3]([O-])([O-])[O-]', 'Cc1ccccc1', 'CCCCCC', 'O=C([O-])O.[Na+]', 'CN(C)C=O', 'CNCCNC', 'CO[Na]', 'C(=NC1CCCCC1)=NC1CCCCC1', 'CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F', 'B1C2CCCC1CCC2', 'CCOC(=O)Cl', 'CCOCC', 'CN(C)c1ccccn1', 'O=C(Cl)C(=O)Cl', '[Li]C(C)CC', 'C[Si](C)(C)[N-][Si](C)(C)C.[Na+]', 'CC(C)C[Al+]CC(C)C.[H-]', '[Ca+2].[Cl-].[Cl-]', 'BrB(Br)Br', '[F-].[K+]', 'CCN(C(C)C)C(C)C', 'C1CCNC1', 'CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21', 'CO', 'B', 'CC[N+](CC)(CC)S(=O)(=O)N=C([O-])OC', 'C[O-].[Na+]', 'O=[Cr](=O)=O', 'CSC', 'CCCCC', 'CN[C@@H]1CCCC[C@H]1NC', 'CC[SiH](CC)CC', 'COc1nc(OC)nc([N+]2(C)CCOCC2)n1.[Cl-]', 'CC#N', 'C[n+]1ccccc1Cl.[I-]', 'Cl[Sn](Cl)(Cl)Cl', 'CN1CCOCC1', 'CC(C)OC(=O)N=NC(=O)OC(C)C', 'Cc1ccc(S(=O)(=O)[O-])cc1.c1cc[nH+]cc1', 'C1CCNCC1', 'CC(C)[N-]C(C)C.[Li+]', 'CCN=C=NCCCN(C)C', 'O=S(=O)([O-])[O-].[Na+].[Na+]', 'O=C(OOC(=O)c1ccccc1)c1ccccc1', '[Li]CCCC', 'O=C([O-])[O-].[Cs+].[Cs+]', '[Cl-].[Li+]', 'C[N+]1([O-])CCOCC1', 'FB(F)F', 'O=C(O)[C@@H]1CCCN1', 'CC(C)NC(C)C', '[Al]', 'CN(C)c1ccccc1', 'O=C(OC(=O)C(F)(F)F)C(F)(F)F', 'O=S([O-])[O-].[Na+].[Na+]', 'O=P(O)(O)O', 'NCCN', 'COCCOC', '[Cs+].[F-]', 'CC(=O)[O-].[K+]', 'CC(=O)OC(C)=O', 'CCCCCCC', 'CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2', 'Cl[Sn]Cl', 'CN1CC[NH+](C)C1Cl.[Cl-]', 'CC(C)(C)O', 'BrBr', 'CN1CCCC1=O', '[Cl-].[Na+]', 'CC(C)O', 'II', 'O=C1OCCN1P(=O)(Cl)N1CCOC1=O', 'CS(C)=O', '[H-].[Na+]', 'C[Si](C)(C)Cl', 'C[Si](C)(C)I', "", 'CN(C)C(=N)N(C)C', 'O=[Cr](=O)([O-])O[Cr](=O)(=O)[O-].c1cc[nH+]cc1.c1cc[nH+]cc1', 'CC(=O)O', 'CS(=O)(=O)O', 'O=C([O-])[O-].[Ca+2]', 'Cl', 'O=O', 'O=[Cr](=O)([O-])Cl.c1cc[nH+]cc1', 'CCN(CC)CC', 'O=S([O-])([O-])=S.[Na+].[Na+]', 'Cc1ccccc1P(c1ccccc1C)c1ccccc1C', '[Na]', 'O=S([O-])O.[Na+]', 'CC(C)(C)O[K]', 'Cc1cccc(C)n1', 'CC(C)=O', 'CC(C)N=C=NC(C)C', 'CCNCC', 'C[Si](C)(C)Br', 'Cc1ccc(S(=O)(=O)Cl)cc1', 'CC(C)(C)OC(=O)N=NC(=O)OC(C)(C)C', 'CCCC[Sn](=O)CCCC', 'CC(=O)N(C)C', 'CCCC[SnH](CCCC)CCCC', 'CC(C)(C)O[Na]', '[Li+].[OH-]', 'CCOC(=O)N=NC(=O)OCC', 'C[Si](C)(C)[N-][Si](C)(C)C.[K+]', 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1', 'O=S(=O)(O)O', 'O=C([O-])[O-].[Na+].[Na+]', 'C1CC[NH2+]CC1.CC(=O)[O-]', '[K]', 'C1COCCN1']

def get_conformer(mol):
    mol = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, useRandomCoords = True, maxAttempts = 10) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    mol = AllChem.RemoveHs(mol)
    return mol

def get_reaction_center_and_mapping(reaction,reactants,products):
    reactant_center_atoms = reaction.GetReactingAtoms()

    reactant_offsets = [0]
    for reactant_molecule in reaction.GetReactants()[:-1]:
        reactant_offsets.append(reactant_offsets[-1] + reactant_molecule.GetNumAtoms())

    reactant_center_atoms = [reactant_atom + reactant_offset
                            for reactant_offset,reactant_atoms 
                            in zip(reactant_offsets,reactant_center_atoms)
                            for reactant_atom 
                            in reactant_atoms]
    
    product_center_atoms = []
    for reactant_center_atom in reactant_center_atoms:
        map_number = reactants.GetAtomWithIdx(reactant_center_atom).GetAtomMapNum()
        for product_atom_index,product_atom in enumerate(products.GetAtoms()):
            if product_atom.GetAtomMapNum() == map_number:
                product_center_atoms.append(product_atom_index)
                break
    
    for atom_index,atom in enumerate(reactants.GetAtoms()):
        if atom.GetAtomMapNum() == 0:
            reactant_center_atoms.append(atom_index)

    for atom_index,atom in enumerate(products.GetAtoms()):
        if atom.GetAtomMapNum() == 0:
            product_center_atoms.append(atom_index)
    
    reactant_map_index = [0 for _ in range(len(reactants.GetAtoms()))]
    product_map_index = [0 for _ in range(len(products.GetAtoms()))]

    for reactant_atom in reactants.GetAtoms():
        map_number = reactant_atom.GetAtomMapNum()
        for product_atom in products.GetAtoms():
            if product_atom.GetAtomMapNum() == map_number:
                reactant_map_index[reactant_atom.GetIdx()] = product_atom.GetIdx() + 1
                product_map_index[product_atom.GetIdx()] = reactant_atom.GetIdx() + 1
                break
    
    reactant_center_atoms = list(set(reactant_center_atoms))
    product_center_atoms = list(set(product_center_atoms))
    return reactant_center_atoms,product_center_atoms,reactant_map_index,product_map_index

def get_donor_and_acceptor_info(mol):
    donor_info, acceptor_info = [], []
    for feat in chem_feature_factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == 'Donor': donor_info.append(feat.GetAtomIds()[0])
        if feat.GetFamily() == 'Acceptor': acceptor_info.append(feat.GetAtomIds()[0])
    
    return donor_info, acceptor_info

def get_chirality_info(atom):
    return [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] if atom.HasProp('Chirality') else [0, 0]
    
def get_stereochemistry_info(bond):
    return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] if bond.HasProp('Stereochemistry') else [0, 0]    

def initialize_stereo_info(mol):
    for element in Chem.FindPotentialStereo(mol):
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': 
            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': 
            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    return mol

reactant_mol_block = ""
product_mol_block = ""

def get_atom_3d_info(mol):
    node_positions = torch.tensor(mol.GetConformer().GetPositions())
    return node_positions

def get_atom_attribute(mol):
    mol = initialize_stereo_info(mol)
    donor_list, acceptor_list = get_donor_and_acceptor_info(mol) 
    atom_feature1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_feature2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_feature4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_feature5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_feature6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-2]
    atom_feature7 = np.array([[(j in donor_list), (j in acceptor_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_feature8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    atom_feature9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_feature10 = np.array([get_chirality_info(a) for a in mol.GetAtoms()], dtype = bool)
    attribute = np.hstack([ atom_feature1, 
                            atom_feature2, 
                            atom_feature3, 
                            atom_feature4, 
                            atom_feature5, 
                            atom_feature6, 
                            atom_feature7, 
                            atom_feature8, 
                            atom_feature9, 
                            atom_feature10])
    attribute = torch.tensor(attribute)
    return attribute

def get_atom_fingerprint(mol,fingerprint_radius = 2):
    node_fingerprint = [[] for _ in range(mol.GetNumAtoms())]
    info = {}
    AllChem.GetMorganFingerprint(mol, fingerprint_radius, bitInfo=info)
    for fingerprint_index,value in info.items():
        for atom_index,_ in value:
            node_fingerprint[atom_index].append(fingerprint_index)
    return node_fingerprint

def get_molecule_fingerprint(mols,fingerprint_radius = 2):
    attributes = []
    for mol in mols:
        info = {}
        mol.UpdatePropertyCache()
        Chem.GetSSSR(mol)
        AllChem.GetMorganFingerprint(mol, fingerprint_radius, bitInfo=info)
        attributes.append(list(info.keys()))
    return attributes

def get_atom_bond_attribute(mol):
    bond_feature1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
    bond_feature2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
    bond_feature3 = np.array([get_stereochemistry_info(b) for b in mol.GetBonds()], dtype = bool)
    edge_attribute = np.hstack([bond_feature1, bond_feature2, bond_feature3])
    edge_attribute = torch.tensor(edge_attribute)
    return edge_attribute

def get_reaction_graph(smiles = ""):
    reactants,products = smiles.split(">>")
    reactants = Chem.MolFromSmiles(reactants)
    products = Chem.MolFromSmiles(products)

    if reactants is None or products is None:
        raise Exception("Invalid SMILES string")

    reactants = get_conformer(reactants)
    products = get_conformer(products)

    if reactants is None or products is None:
        raise Exception("Conformer generation failed")

    reaction = rdChemReactions.ReactionFromSmarts(smiles)
    reaction.Initialize()

    (reactant_center_atoms,
     product_center_atoms,
     reactant_map_index,
     product_map_index) = get_reaction_center_and_mapping(reaction,reactants,products)
    
    num_atom_nodes = reactants.GetNumAtoms() + products.GetNumAtoms()
    num_molecule_nodes = len(reaction.GetReactants()) + len(reaction.GetProducts())
    num_nodes = num_atom_nodes + num_molecule_nodes

    reactant_attribute = get_atom_attribute(reactants)
    product_attribute = get_atom_attribute(products)

    reactant_fingerprint = get_atom_fingerprint(reactants)
    product_fingerprint = get_atom_fingerprint(products)

    global reactant_mol_block, product_mol_block
    reactant_mol_block = Chem.MolToMolBlock(reactants)
    product_mol_block = Chem.MolToMolBlock(products)
    reactant_positions = get_atom_3d_info(reactants)
    product_positions = get_atom_3d_info(products)
    
    node_attribute_padding = torch.stack([torch.zeros_like(reactant_attribute[0]) ]* num_molecule_nodes)
    node_positions_padding = torch.stack([torch.zeros_like(reactant_positions[0]) ]* num_molecule_nodes)

    reactant_molecule_fingerprint = get_molecule_fingerprint(reaction.GetReactants())
    product_molecule_fingerprint = get_molecule_fingerprint(reaction.GetProducts())

    node_attribute = torch.cat([reactant_attribute,product_attribute,node_attribute_padding],dim=0)
    node_positions = torch.cat([reactant_positions,product_positions,node_positions_padding],dim=0)

    node_type = torch.zeros([num_nodes,4], dtype = torch.uint8)
    node_type[:reactants.GetNumAtoms(),0] = 1
    node_type[reactants.GetNumAtoms():num_atom_nodes,1] = 1
    node_type[num_atom_nodes:num_atom_nodes + len(reaction.GetReactants()),2] = 1
    node_type[num_atom_nodes + len(reaction.GetReactants()):,3] = 1

    reactant_center_atoms = np.array(reactant_center_atoms)
    product_center_atoms = np.array(product_center_atoms) + reactants.GetNumAtoms()

    node_center = torch.zeros([num_nodes,2],dtype = torch.uint8)
    node_center[reactant_center_atoms,0] = 1
    node_center[product_center_atoms,1] = 1

    src = np.empty(0).astype(int)
    dst = np.empty(0).astype(int)
    message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    message_graph.ndata['attribute'] = node_attribute.to(torch.uint8)
    message_graph.ndata['position'] = node_positions.to(torch.float32)
    message_graph.ndata['center'] = node_center.to(torch.uint8)
    message_graph.ndata['type'] = node_type.to(torch.uint8)
    
    reactant_bond_attribute = get_atom_bond_attribute(reactants)
    reactant_bond_attribute = torch.cat([reactant_bond_attribute,reactant_bond_attribute],dim=0)
    reactant_bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in reactants.GetBonds()], dtype=int)
    src = np.hstack([reactant_bonds[:,0], reactant_bonds[:,1]])
    dst = np.hstack([reactant_bonds[:,1], reactant_bonds[:,0]])
    reactant_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reactant_message_passing_graph.edata['attribute'] = reactant_bond_attribute.to(torch.uint8)

    product_bond_attribute = get_atom_bond_attribute(products)
    product_bond_attribute = torch.cat([product_bond_attribute,product_bond_attribute],dim=0)
    product_bonds = np.array([[b.GetBeginAtomIdx() + reactants.GetNumAtoms(), b.GetEndAtomIdx() + reactants.GetNumAtoms()] for b in products.GetBonds()], dtype=int)
    src = np.hstack([product_bonds[:,0], product_bonds[:,1]])
    dst = np.hstack([product_bonds[:,1], product_bonds[:,0]])
    product_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    product_message_passing_graph.edata['attribute'] = product_bond_attribute.to(torch.uint8)
    
    offset = 0
    edges = []
    for index,molecule in enumerate(list(reaction.GetReactants())):
        edges += [(i + offset,num_atom_nodes + index) for i in range(molecule.GetNumAtoms())]
        offset += molecule.GetNumAtoms()
    edges = np.array(edges)
    aggregate_edge_attribute = np.zeros_like(edges)
    disperse_edge_attribute = np.zeros_like(edges)
    aggregate_edge_attribute[:,0] = 1
    disperse_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([aggregate_edge_attribute,disperse_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    reactant_molecule_message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reactant_molecule_message_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    edges = []
    for index,molecule in enumerate(list(reaction.GetProducts())):
        edges += [(i + offset,num_atom_nodes + len(reaction.GetReactants()) + index) for i in range(molecule.GetNumAtoms())]
        offset += molecule.GetNumAtoms()
    edges = np.array(edges)
    aggregate_edge_attribute = np.zeros_like(edges)
    disperse_edge_attribute = np.zeros_like(edges)
    aggregate_edge_attribute[:,0] = 1
    disperse_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([aggregate_edge_attribute,disperse_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    product_molecule_message_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    product_molecule_message_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    assert 0 not in product_map_index, "Product map index should not contain 0"
    product_map_index = np.array(product_map_index) - 1
    product_index = np.arange(0,len(product_map_index)) + reactants.GetNumAtoms()
    edges = np.concatenate([product_map_index[:,np.newaxis],product_index[:,np.newaxis]],axis=-1)
    forward_edge_attribute = np.zeros_like(edges)
    backward_edge_attribute = np.zeros_like(edges)
    forward_edge_attribute[:,0] = 1
    backward_edge_attribute[:,1] = 1
    edge_attribute = np.concatenate([forward_edge_attribute,backward_edge_attribute],axis=0)
    src = np.hstack([edges[:,0], edges[:,1]])
    dst = np.hstack([edges[:,1], edges[:,0]])
    reaction_atom_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reaction_atom_message_passing_graph.edata['attribute'] = torch.tensor(edge_attribute).to(torch.uint8)

    forward_edges = []
    backward_edges = []
    for reactant_molecule_index in range(len(reaction.GetReactants())):
        for product_molecule_index in range(len(reaction.GetProducts())):
            forward_edges.append((reactant_molecule_index + num_atom_nodes, product_molecule_index + num_atom_nodes + len(reaction.GetReactants())))
            backward_edges.append((product_molecule_index + num_atom_nodes + len(reaction.GetReactants()), reactant_molecule_index + num_atom_nodes))
    edges = forward_edges + backward_edges
    for reactant_molecule_index_1 in range(len(reaction.GetReactants())):
        for reactant_molecule_index_2 in range(len(reaction.GetReactants())):
            if reactant_molecule_index_1 != reactant_molecule_index_2:
                edges.append((reactant_molecule_index_1 + num_atom_nodes,reactant_molecule_index_2 + num_atom_nodes))
    for product_molecule_index_1 in range(len(reaction.GetProducts())):
        for product_molecule_index_2 in range(len(reaction.GetProducts())):
            if product_molecule_index_1 != product_molecule_index_2:
                edges.append((product_molecule_index_1 + num_atom_nodes + len(reaction.GetReactants()),product_molecule_index_2+ num_atom_nodes + len(reaction.GetReactants())))

    edge_attribute = torch.zeros([len(edges),4])
    edge_attribute[:len(forward_edges),0] = 1
    edge_attribute[len(forward_edges):2*len(forward_edges),1] = 1
    edge_attribute[2*len(forward_edges):2*len(forward_edges) + len(reaction.GetReactants())**2 - len(reaction.GetReactants()),2] = 1
    edge_attribute[2*len(forward_edges) + len(reaction.GetReactants())**2 - len(reaction.GetReactants()):,3] = 1
    edges = np.array(edges)
    src = edges[:,0]
    dst = edges[:,1]
    reaction_molecule_message_passing_graph = dgl.graph((src, dst), num_nodes = num_nodes)
    reaction_molecule_message_passing_graph.edata['attribute'] = edge_attribute.to(torch.uint8)

    fingerprint = reactant_fingerprint + product_fingerprint + reactant_molecule_fingerprint + product_molecule_fingerprint
    
    return {"message_graph":message_graph,
            "reactant_message_passing_graph":reactant_message_passing_graph,
            "product_message_passing_graph":product_message_passing_graph,
            "reactant_molecule_message_graph":reactant_molecule_message_graph,
            "product_molecule_message_graph":product_molecule_message_graph,
            "reaction_atom_message_passing_graph":reaction_atom_message_passing_graph,
            "reaction_molecule_message_passing_graph":reaction_molecule_message_passing_graph,
            "fingerprint":fingerprint}

parser = ArgumentParser(description='hyperparameters')
parser.add_argument('--seed', type=float,default=123)
parser.add_argument('--device', type=str,default="1")
parser.add_argument('--dataset', type=str,default="../dataset")
parser.add_argument('--model_dir', type=str,default="./")
parser.add_argument("--reactant",type=str,default="")
parser.add_argument("--reagent",type=str,default="")
parser.add_argument("--product",type=str,default="")
parser.add_argument("--workdir",type=str,default="")

parser.add_argument("--pretrain_epoches", type=int, default=50)
parser.add_argument("--finetune_epoches", type=int, default=100)

parser.add_argument("--learning_rate_pretrain", type=float, default=5e-4)
parser.add_argument("--weight_decay_pretrain", type=float, default=1e-10)
parser.add_argument("--learning_rate_schedule_pretrain", type=str,default="min") 
parser.add_argument("--learning_rate_factor_pretrain",type=float,default=0.1)
parser.add_argument("--learning_rate_patience_pretrain",type=int,default=5)
parser.add_argument("--min_learning_rate_pretrain",type=float,default=1e-8)
parser.add_argument("--learning_rate_verbose_pretrain",type=bool,default=True)

parser.add_argument("--learning_rate_finetune", type=float, default=5e-4)
parser.add_argument("--weight_decay_finetune", type=float, default=1e-10)
parser.add_argument("--learning_rate_schedule_finetune", type=str,default="min") 
parser.add_argument("--learning_rate_factor_finetune",type=float,default=0.1)
parser.add_argument("--learning_rate_patience_finetune",type=int,default=5)
parser.add_argument("--min_learning_rate_finetune",type=float,default=1e-8)
parser.add_argument("--learning_rate_verbose_finetune",type=bool,default=True)

parser.add_argument("--dim_hidden", type=int, default=4096)
parser.add_argument("--dim_node_attribute", type=int,default = 110)
parser.add_argument("--dim_edge_attribute", type=int,default = 13)
parser.add_argument("--dim_edge_length", type=int,default = 16)
parser.add_argument("--dim_hidden_features", type=int,default = 200)
parser.add_argument("--message_passing_step", type=int,default = 3)
parser.add_argument("--pooling_step", type=int,default = 2)
parser.add_argument("--num_layers_pooling", type=int,default = 2)

parser.add_argument("--load_last",type=bool,default=False)
parser.add_argument("--smoothing_catalyst1",type = float,default = 1)
parser.add_argument("--smoothing_solvent1",type = float,default = 1)
parser.add_argument("--smoothing_solvent2",type = float,default = 1)
parser.add_argument("--smoothing_reagent1",type = float,default = 1)
parser.add_argument("--smoothing_reagent2",type = float,default = 1)

parser.add_argument("--none_weight_catalyst1",type = float,default = 1)
parser.add_argument("--none_weight_solvent1",type = float,default = 1)
parser.add_argument("--none_weight_solvent2",type = float,default = 1)
parser.add_argument("--none_weight_reagent1",type = float,default = 1)
parser.add_argument("--none_weight_reagent2",type = float,default = 1)

parser.add_argument("--save_delta", type=int,default = 10)
parser.add_argument("--accumulation_steps", type=int,default = 4)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
dgl.random.seed(seed)
torch.backends.cudnn.deterministic = True

model = ReactionGraphModel(dataset_train = None,
                          dataset_test = None,
                          dataset_val = None,
                          model_dir = args.model_dir,
                          parameters_for_model = {
                              "dim_hidden":args.dim_hidden,
                              "dim_node_attribute":args.dim_node_attribute,
                              "dim_edge_attribute":args.dim_edge_attribute,
                              "dim_edge_length":args.dim_edge_length,
                              "dim_hidden_features":args.dim_hidden_features,
                              "message_passing_step":args.message_passing_step,
                              "pooling_step":args.pooling_step,
                              "num_layers_pooling":args.num_layers_pooling},
                          parameters_for_optimizer_pretrain = {
                              "lr":args.learning_rate_pretrain, 
                              "weight_decay":args.weight_decay_pretrain},
                          parameters_for_scheduler_pretrain = {
                              "mode":args.learning_rate_schedule_pretrain, 
                              "factor":args.learning_rate_factor_pretrain, 
                              "patience":args.learning_rate_patience_pretrain, 
                              "min_lr":args.min_learning_rate_pretrain, 
                              "verbose":args.learning_rate_verbose_pretrain},
                          parameters_for_optimizer_finetune = {
                              "lr":args.learning_rate_finetune, 
                              "weight_decay":args.weight_decay_finetune},
                          parameters_for_scheduler_finetune = {
                              "mode":args.learning_rate_schedule_finetune, 
                              "factor":args.learning_rate_factor_finetune, 
                              "patience":args.learning_rate_patience_finetune, 
                              "min_lr":args.min_learning_rate_finetune, 
                              "verbose":args.learning_rate_verbose_finetune},
                          none_weights = {
                              "catalyst1":args.none_weight_catalyst1,
                              "solvent1":args.none_weight_solvent1,
                              "solvent2":args.none_weight_solvent2,
                              "reagent1":args.none_weight_reagent1,
                              "reagent2":args.none_weight_reagent2},
                          smoothing=[
                              args.smoothing_catalyst1,
                              args.smoothing_solvent1,
                              args.smoothing_solvent2,
                              args.smoothing_reagent1,
                              args.smoothing_reagent2],
                          accumulation_steps=args.accumulation_steps)

model.load("uspto_condition.ckpt")

def construct(g):
    num_nodes = g.num_nodes()
    batch_num_nodes = g.batch_num_nodes()
    base = 0
    batch_num_edges = []
    new_edges = set()
    last_num_edges = 0
    for current_batch_num_nodes in batch_num_nodes:
        for offset in range(current_batch_num_nodes):
            i = base + offset
            predecessors = g.predecessors(i)
            for j in predecessors:
                for k in predecessors:
                    if j!=k:
                        new_edges.add((j.item(),k.item()))
        base += current_batch_num_nodes
        current_batch_num_edges = len(new_edges) - last_num_edges
        last_num_edges = len(new_edges)
        batch_num_edges.append(current_batch_num_edges)

    batch_num_edges = torch.tensor(batch_num_edges)
    src,dst = list(zip(*new_edges))
    src = np.array(src)
    dst = np.array(dst)
    new_g = dgl.graph((src,dst),num_nodes=num_nodes)
    new_g.set_batch_num_nodes(batch_num_nodes)
    new_g.set_batch_num_edges(batch_num_edges)
    return new_g

reactant = args.reactant
product = args.product

smiles = reactant + ">>" + product
mapper = RXNMapper()
mapped_smiles = mapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn']

workdir = args.workdir
with open(f"{workdir}/mapped_smiles.json","w") as f:
    json.dump({"smiles":mapped_smiles},f)
inputs = get_reaction_graph(mapped_smiles)

with open(f"{workdir}/coordinates.json","w") as f:
    json.dump({
        "reactant":reactant_mol_block,
        "product":product_mol_block
    },f)

message_graph = dgl.batch([inputs["message_graph"]])
reactant_message_passing_graph = dgl.batch([inputs["reactant_message_passing_graph"]])
product_message_passing_graph = dgl.batch([inputs["product_message_passing_graph"]])
reaction_atom_message_passing_graph = dgl.batch([inputs["reaction_atom_message_passing_graph"]])
reactant_geometry_message_graph = construct(reactant_message_passing_graph)
product_geometry_message_graph = construct(product_message_passing_graph)
positions = message_graph.ndata["position"]

def get_direction(graph):
    src = graph.edges()[0]
    dst = graph.edges()[1]
    directions = positions[dst] - positions[src]
    graph.edata["direction"] = directions

get_direction(reactant_message_passing_graph)
get_direction(product_message_passing_graph)
get_direction(reactant_geometry_message_graph)
get_direction(product_geometry_message_graph)

num_edges_each_batch = 0
num_edges_each_batch += reactant_message_passing_graph.batch_num_edges()
num_edges_each_batch += product_message_passing_graph.batch_num_edges()
num_edges_each_batch += reactant_geometry_message_graph.batch_num_edges()
num_edges_each_batch += product_geometry_message_graph.batch_num_edges()
num_edges_each_batch += reaction_atom_message_passing_graph.batch_num_edges()

reactant_atom_edge_attribute = reactant_message_passing_graph.edata["attribute"]
product_atom_edge_attribute = product_message_passing_graph.edata["attribute"]
reaction_atom_edge_attribute = reaction_atom_message_passing_graph.edata["attribute"]

reactant_atom_num_edge = reactant_message_passing_graph.num_edges()
product_atom_num_edge = product_message_passing_graph.num_edges()
reactant_geometry_atom_num_edge = reactant_geometry_message_graph.num_edges()
product_geometry_atom_num_edge = product_geometry_message_graph.num_edges()
reaction_atom_num_edge = reaction_atom_message_passing_graph.num_edges()

dim_edge_attribute = reactant_atom_edge_attribute.size(1)

reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5])
reactant_atom_padding[:,0] = 1
reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

product_atom_padding = torch.zeros([product_atom_num_edge,5])
product_atom_padding[:,1] = 1
product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

reactant_geometry_atom_edge_attribute = torch.zeros([reactant_geometry_atom_num_edge,dim_edge_attribute + 5])
reactant_geometry_atom_edge_attribute[:,-1] = 1

product_geometry_atom_edge_attribute = torch.zeros([product_geometry_atom_num_edge,dim_edge_attribute + 5])
product_geometry_atom_edge_attribute[:,-1] = 1

reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2])
reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1])
reaction_atom_edge_attribute = torch.cat([reaction_atom_padding_left,reaction_atom_edge_attribute,reaction_atom_padding_right],dim = -1)

reactant_message_passing_graph.edata["attribute"] = reactant_atom_edge_attribute
product_message_passing_graph.edata["attribute"] = product_atom_edge_attribute
reactant_geometry_message_graph.edata["attribute"] = reactant_geometry_atom_edge_attribute
product_geometry_message_graph.edata["attribute"] = product_geometry_atom_edge_attribute
reaction_atom_message_passing_graph.edata["attribute"] = reaction_atom_edge_attribute

reactant_message_passing_graph.edata['length'] = reactant_message_passing_graph.edata['direction'].norm(dim=-1)
del reactant_message_passing_graph.edata['direction']
product_message_passing_graph.edata['length'] = product_message_passing_graph.edata['direction'].norm(dim=-1)
del product_message_passing_graph.edata['direction']
reactant_geometry_message_graph.edata['length'] = reactant_geometry_message_graph.edata['direction'].norm(dim=-1)
del reactant_geometry_message_graph.edata['direction']
product_geometry_message_graph.edata['length'] = product_geometry_message_graph.edata['direction'].norm(dim=-1)
del product_geometry_message_graph.edata['direction']
reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge]).float()

message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reactant_geometry_message_graph,product_geometry_message_graph,reaction_atom_message_passing_graph])

node_type = message_graph.ndata["type"]
select_reactant_atoms = node_type[:,0] == 1
select_product_atoms = node_type[:,1] == 1
select_reactant_molecule = node_type[:,2] == 1
select_product_molecule = node_type[:,3] == 1
select_atoms = select_reactant_atoms | select_product_atoms
select_molecule = select_reactant_molecule | select_product_molecule


def get_num_node_each_batch(vector):
    count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
    count_list = torch.tensor(count_list)
    return count_list

num_atoms_each_batch = get_num_node_each_batch(select_atoms)

indices = torch.arange(node_type.size(0))
remove_indices = indices[select_molecule]

message_graph.remove_nodes(remove_indices)
message_passing_graph.remove_nodes(remove_indices)
message_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
message_passing_graph.set_batch_num_edges(num_edges_each_batch)

inputs = {}
inputs["message_graph"] = message_graph
inputs["message_passing_graph"] = message_passing_graph

message_graph = inputs["message_graph"].to("cuda")
message_passing_graph = inputs["message_passing_graph"].to("cuda")
message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
inputs = {"message_graph":message_graph,
          "message_passing_graph":message_passing_graph}
result = model.inference(inputs)[0]

attention_weight = attention_weight.ravel().tolist()
with open(f"{workdir}/attention_weights.json","w") as f:
    json.dump({"weight":attention_weight},f)

conditions = []
for row in result:
    condition = [
        catalysts[row[0]],
        solvents[row[1]],
        solvents[row[2]],
        reagents[row[3]],
        reagents[row[4]]
    ]
    conditions.append(condition)

for condition in conditions:
    print(condition)    

result = {"conditions":conditions}

workdir = args.workdir
with open(f"{workdir}/result.json","w") as f:
    json.dump(result,f)