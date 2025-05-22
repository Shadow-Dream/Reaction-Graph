import torch
from torch import nn
from torch.nn import Module
from dgl.nn.pytorch import NNConv, Set2Set
from networks.rbf import RBFLayer
import dgl

class DimeReactionNN(Module):
    def __init__(self,
                 dim_node_attribute = 110,
                 dim_edge_attribute = 8,
                 dim_edge_length = 8,
                 dim_hidden_features = 64,
                 dim_hidden = 4096,
                 message_passing_step = 4,
                 pooling_step = 3,
                 num_layers_pooling = 2,
                 dim_edge_edge_angle = 8,
                 dim_edge_edge_length = 8):
        
        super(DimeReactionNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(
            nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
        self.rbf = RBFLayer(dim_edge_length)
        self.edge_rbf = RBFLayer(dim_edge_edge_length)
        self.edge_sbf = RBFLayer(dim_edge_edge_angle)
        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
        self.edge_function = nn.Linear(dim_edge_edge_length + dim_edge_edge_angle, dim_hidden_features * dim_hidden_features)
        self.edge_gnn = NNConv(dim_hidden_features, dim_hidden_features, self.edge_function, 'sum',bias=False)

        self.bond_update = nn.Sequential(nn.Linear(dim_hidden_features,dim_hidden_features), nn.ReLU())
        self.angle_update = nn.Sequential(nn.Linear(dim_hidden_features,dim_hidden_features), nn.ReLU())
    
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)

        self.pooling = Set2Set(input_dim = dim_hidden_features * 2,
                               n_iters = pooling_step,
                               n_layers = num_layers_pooling)

        self.sparsify = nn.Sequential(
            nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU()
        )

        self.activation = nn.ReLU()
        
        self.message_passing_step = message_passing_step

    def forward(self, inputs):
        message_graph = inputs["message_graph"]
        message_passing_graph = inputs["message_passing_graph"]
        edge_message_passing_graph = inputs["edge_message_passing_graph"]

        node_attribute = message_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        
        edge_attribute = message_passing_graph.edata["attribute"]
        edge_length = self.rbf(message_passing_graph.edata["length"])

        edge_edge_length = self.edge_rbf(message_passing_graph.edata["length"])
        edge_edge_length = edge_edge_length[edge_message_passing_graph.edges()[0]]
        edge_edge_angle = edge_message_passing_graph.edata["angle"]
        edge_edge_angle = self.edge_sbf(edge_edge_angle)
        edge_edge_features = torch.cat([edge_edge_angle,edge_edge_length],dim = -1)

        edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            with message_passing_graph.local_scope():
                bond_message = node_features[message_passing_graph.edges()[0]]
                with edge_message_passing_graph.local_scope():
                    edge_message_passing_graph.srcdata['h'] = self.angle_update(bond_message).unsqueeze(-1)
                    edge_message_passing_graph.edata['w'] = self.edge_gnn.edge_func(edge_edge_features).view(-1, self.edge_gnn._in_src_feats, self.edge_gnn._out_feats)
                    edge_message_passing_graph.update_all(dgl.function.u_mul_e('h', 'w', 'm'), self.edge_gnn.reducer('m', 'neigh'))
                    angle_message = edge_message_passing_graph.dstdata['neigh'].sum(dim=1)
                message_passing_graph.edata['h'] = self.bond_update(bond_message) + angle_message
                message_passing_graph.edata['h'] = message_passing_graph.edata['h'].unsqueeze(-1)
                message_passing_graph.edata['w'] = self.gnn.edge_func(edge_features).view(-1, self.gnn._in_src_feats, self.gnn._out_feats)
                message_passing_graph.edata['m'] = message_passing_graph.edata['h'] * message_passing_graph.edata['w']
                message_passing_graph.edata['m'] = message_passing_graph.edata['m'].sum(dim=1)
                message_passing_graph.update_all(dgl.function.copy_e('m', 'new_m'), self.gnn.reducer('new_m', 'neigh'))
                node_features = message_passing_graph.dstdata['neigh'] + self.gnn.bias

            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        reaction_features = self.pooling(message_passing_graph, node_aggregation)
        reaction_features = self.sparsify(reaction_features)
        return reaction_features