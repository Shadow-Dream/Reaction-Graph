import torch
from torch import nn
from torch.nn import Module
from dgl.nn.pytorch import NNConv, Set2Set, SumPooling
from dgl.ops import segment
from networks.components.rbf import RBFLayer

class RBFMPNN(Module):
    def __init__(self,
                 dim_node_attribute = 110,
                 dim_edge_attribute = 8,
                 dim_edge_length = 8,
                 dim_hidden_features = 64,
                 dim_hidden = 4096,
                 message_passing_step = 4,
                 pooling_step = 3,
                 num_layers_pooling = 2,
                 aggregation_level = "reaction",
                 aggregation_method = "set2set",
                 **kwargs):
        super(RBFMPNN, self).__init__()
        
        self.atom_feature_projector = nn.Sequential(nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
       
        self.dim_edge_length = dim_edge_length
        if dim_edge_length:
            self.rbf = RBFLayer(dim_edge_length)

        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)

        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)
        self.activation = nn.ReLU()

        self.message_passing_step = message_passing_step

        self.aggregation_level = aggregation_level
        self.aggregation_method = aggregation_method

        if aggregation_method == "set2set":
            self.pooling = Set2Set(input_dim = dim_hidden_features * 2,
                                n_iters = pooling_step,
                                n_layers = num_layers_pooling)
        
            self.sparsify = nn.Sequential(nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU())
        elif aggregation_method == "sumplus":
            self.sparsify = nn.Sequential(nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU())
        elif aggregation_method == "sum":
            self.sparsify = nn.Sequential(nn.Linear(dim_hidden_features * 2, dim_hidden), nn.PReLU())
            

    def forward(self, reaction_graph):
        node_attribute = reaction_graph.ndata["attribute"]
        node_features = self.atom_feature_projector(node_attribute)
        edge_attribute = reaction_graph.edata["attribute"]
        if self.dim_edge_length:
            edge_length = self.rbf(reaction_graph.edata["length"])
            edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        else:
            edge_features = edge_attribute
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(reaction_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        if self.aggregation_level == "reaction":
            if self.aggregation_method == "set2set":
                reaction_features = self.pooling(reaction_graph, node_aggregation)
            elif self.aggregation_method == "sumplus":
                atom_per_reaction = reaction_graph.batch_num_nodes()
                reaction_features = torch.cat([node_aggregation,node_aggregation],dim = -1)
                reaction_features = segment.segment_reduce(atom_per_reaction, reaction_features, reducer="sum")
            elif self.aggregation_method == "sum":
                atom_per_reaction = reaction_graph.batch_num_nodes()
                reaction_features = node_aggregation
                reaction_features = segment.segment_reduce(atom_per_reaction, reaction_features, reducer="sum")
            reaction_features = self.sparsify(reaction_features)
            return reaction_features
        
        elif self.aggregation_level == "molecule":
            atom_per_molecule = reaction_graph.gdata["atom_per_molecule"]
            molecule_per_reaction = reaction_graph.gdata["molecule_per_reaction"]
            if self.aggregation_method == "set2set":
                reaction_graph.set_batch_num_nodes(atom_per_molecule)
                reaction_features = self.pooling(reaction_graph, node_aggregation)
            elif self.aggregation_method == "sumplus":
                reaction_features = torch.cat([node_aggregation,node_aggregation],dim = -1)
                reaction_features = segment.segment_reduce(atom_per_molecule, reaction_features, reducer="sum")
            elif self.aggregation_method == "sum":
                reaction_features = node_aggregation
                reaction_features = segment.segment_reduce(atom_per_molecule, reaction_features, reducer="sum")
                
            reaction_features = self.sparsify(reaction_features)
            reaction_features = segment.segment_reduce(molecule_per_reaction, reaction_features, reducer="sum")
            feature_dim = reaction_features.shape[-1]
            reaction_features = reaction_features.reshape(-1,3,feature_dim)
            reaction_features[:,1] += reaction_features[:,0]
            reaction_features = reaction_features[:,1:]
            batch_size = reaction_features.shape[0]
            reaction_features = reaction_features.reshape(batch_size,-1)
            return reaction_features