import torch
from torch import nn
from torch.nn import Module
from dgl.nn.pytorch import NNConv, Set2Set
from networks.rbf import RBFLayer

class LeavingGroupNN(Module):
    def __init__(self,
                 dim_node_attribute = 110,
                 dim_edge_attribute = 8,
                 dim_edge_length = 8,
                 dim_hidden_features = 64,
                 dim_hidden = 4096,
                 message_passing_step = 4,
                 pooling_step = 3,
                 num_layers_pooling = 2,
                 type_leaving_groups = 206,
                 dim_type=69):
        
        super(LeavingGroupNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(
            nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
        self.rbf = RBFLayer(dim_edge_length)
        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
    
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)

        self.leaving_group_classification = nn.Sequential(
            nn.Linear(2 * dim_hidden_features, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, type_leaving_groups)
        )

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step

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
        node_leaving_group = self.leaving_group_classification(node_aggregation)
        return node_leaving_group