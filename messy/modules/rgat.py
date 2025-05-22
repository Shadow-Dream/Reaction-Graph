import dgl
import torch
import torch.nn.functional as func
from dgl.nn.pytorch import GlobalAttentionPooling,EGATConv,NNConv,Set2Set
import numpy as np
from torch import nn

class RGAT(torch.nn.Module):
    def __init__(self, dim_node_attribute, dim_edge_attribute, dim_node_hidden_features = 256, dim_edge_hidden_features = 64, dim_hidden = 4096, message_passing_step = 3, num_heads = 4):
        super(RGAT, self).__init__()
        self.node_embedding = nn.Sequential(nn.Linear(dim_node_attribute, dim_node_hidden_features), nn.ReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(dim_edge_attribute, dim_edge_hidden_features), nn.ReLU())
        self.gnns = nn.ModuleList([EGATConv(dim_node_hidden_features, dim_edge_hidden_features, dim_node_hidden_features, dim_edge_hidden_features,  num_heads)] * message_passing_step)
        self.pooling = GlobalAttentionPooling(nn.Linear(dim_node_hidden_features, 1))
        self.sparsify = nn.Linear(dim_node_hidden_features, dim_hidden)

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]
        node_attribute = message_graph.ndata["attribute"]
        edge_attribute = message_passing_graph.edata["attribute"]
        node_features = self.node_embedding(node_attribute)
        edge_features = self.edge_embedding(edge_attribute)
        for gnn in self.gnns:
            node_features, edge_features = gnn(message_passing_graph, node_features, edge_features)
            node_features = node_features.mean(1)
            edge_features = edge_features.mean(1)
        node_features = self.pooling(message_passing_graph, node_features)
        node_features = self.sparsify(node_features)
        return node_features