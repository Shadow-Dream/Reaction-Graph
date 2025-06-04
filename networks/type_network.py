import torch
from torch import nn
from torch.nn import Module
from networks.components.rbf_mpnn import RBFMPNN
from torch.nn import functional as func
    
class TypeNetwork(Module):
    def __init__(self, 
                 dim_types = 1000,
                 dim_hidden = 512,
                 graph_type = "reaction_graph",
                 **parameters_for_reaction_nn):
        super(TypeNetwork, self).__init__()

        if graph_type == "reaction_graph":
            self.reaction_nn = RBFMPNN(dim_hidden=dim_hidden,**parameters_for_reaction_nn)

        self.output = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.PReLU(),
            nn.Linear(dim_hidden, dim_types)
        )

    def embedding(self,inputs):
        features = self.reaction_nn(inputs)
        return features

    def forward(self, inputs):
        features = self.embedding(inputs)
        outputs = self.output(features)
        return outputs
