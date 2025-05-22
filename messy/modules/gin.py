import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import torch.nn.functional as F

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "add",edge_attribute_classes=[5,3]):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(i, emb_dim) for i in edge_attribute_classes]
        )
        for embedding in self.edge_embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr = None):
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        if edge_attr != None:
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
            edge_embeddings = sum([embedding(edge_attr[:,i]) for i,embedding in enumerate(self.edge_embeddings)])
        else:
            edge_embeddings = torch.zeros((edge_index[0].shape[1],x.shape[-1])).to(x.device)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class AtomGIN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0, 
                 atom_attribute_classes = [120,3],
                 edge_attribute_classes = [5,3]):
        super(AtomGIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(i, emb_dim) for i in atom_attribute_classes]
        )
        for embedding in self.x_embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConv(emb_dim,edge_attribute_classes = edge_attribute_classes))
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        x = sum([embedding(x[:,i]) for i,embedding in enumerate(self.x_embeddings)])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation
    
class FragGIN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0, 
                 frag_attribute_classes = 908,
                 edge_attribute_classes = [5,3]):
        super(FragGIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.embedding = nn.Embedding(frag_attribute_classes,emb_dim)

        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConv(emb_dim,edge_attribute_classes = edge_attribute_classes))
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index):
        x = self.embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        return node_representation