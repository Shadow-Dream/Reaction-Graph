import torch
from torch import nn
from torch.nn import Module
from networks.components.rbf_mpnn import RBFMPNN
from torch.nn import functional as func
    
class YieldNetwork(Module):
    def __init__(self, 
                 dim_hidden = 512,
                 graph_type = "reaction_graph",
                 dim_hidden_regression = 512,
                 dropout = 0.1,
                 **parameters_for_reaction_nn):
        super(YieldNetwork, self).__init__()

        if graph_type == "reaction_graph":
            self.reaction_nn = RBFMPNN(dim_hidden=dim_hidden,**parameters_for_reaction_nn)

        self.output = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, 2)
        )

    def embedding(self,inputs):
        features = self.reaction_nn(inputs)
        return features

    def forward(self, inputs):
        features = self.embedding(inputs)
        outputs = self.output(features)
        mean = outputs[:,0:1]
        logvar = outputs[:,1:2]
        return mean, logvar
