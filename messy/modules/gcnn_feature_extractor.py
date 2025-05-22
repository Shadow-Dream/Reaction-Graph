import torch
from torch import nn
from torch.nn import Module
from networks.rgat import RGAT

class GCNNFeatureExtractor(Module):
    def __init__(self,
                 dim_output = 703,
                 dim_hidden = 512, 
                 num_hidden = 0,
                 **parameters_for_rgat):
        super(GCNNFeatureExtractor, self).__init__()
        self.rgat = RGAT(**parameters_for_rgat)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.rgat.readout_features, dim_hidden),nn.PReLU(),
            *[nn.Linear(dim_hidden, dim_hidden),nn.PReLU()]*num_hidden,
            nn.Linear(dim_hidden,dim_output)
        )

    def forward(self,inputs):
        features = self.rgat(inputs["molecule_graphs"],inputs["reaction_graphs"])
        outputs = self.feature_extractor(features)
        return outputs