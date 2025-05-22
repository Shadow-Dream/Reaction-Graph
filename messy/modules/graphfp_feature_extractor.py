import torch
from torch import nn
from networks.gin import AtomGIN,FragGIN
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
class GraphFPFeatureExtractor(nn.Module):
    def __init__(self,embedding_dim,frag_classes,tree_classes,atom_gin_parameters,frag_gin_parameters):
        super(GraphFPFeatureExtractor,self).__init__()
        self.atom_model = AtomGIN(emb_dim=embedding_dim,**atom_gin_parameters)
        self.frag_model = FragGIN(emb_dim=embedding_dim,**frag_gin_parameters)
        self.atom_contrastive_project_head = nn.Linear(embedding_dim,embedding_dim)
        self.atom_frag_classifier_head = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.Linear(embedding_dim,frag_classes)
        )
        self.atom_tree_classifier_head = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.Linear(embedding_dim,tree_classes)
        )
        self.frag_contrastive_project_head = nn.Linear(embedding_dim,embedding_dim)

    def embedding(self,inputs):
        inputs["atom_node_embedding"] = self.embedding_atom(inputs)
        inputs["frag_node_embedding"] = self.embedding_frag(inputs)
        atom_embedding = global_mean_pool(atom_embedding,inputs["atom_batch"])
        frag_embedding = global_mean_pool(frag_embedding,inputs["node_batch"])
        embedding = torch.cat([atom_embedding,frag_embedding],dim=-1)
        return embedding

    def embedding_atom(self, inputs):
        return self.atom_model(inputs["atom_node_attribute"],inputs["atom_edge"],inputs["atom_edge_attribute"])
    
    def embedding_frag(self, inputs):
        return self.frag_model(inputs["frag_node_attribute"],inputs["frag_edge"])
    
    def classify_frag(self, inputs):
        atom_embedding = self.atom_frag_classifier_head(inputs["atom_node_embedding"])
        output = global_mean_pool(atom_embedding,inputs["atom_batch"])
        return output
    
    def classify_tree(self, inputs):
        atom_embedding = self.atom_tree_classifier_head(inputs["atom_node_embedding"])
        output = global_mean_pool(atom_embedding,inputs["atom_batch"])
        return output
    
    def project_atom(self, inputs):
        atom_embedding = self.atom_contrastive_project_head(inputs["atom_node_embedding"])
        atom_embedding = scatter(atom_embedding, inputs["atom_map"], dim=0, reduce="mean")
        return atom_embedding

    def project_frag(self,inputs):
        frag_embedding = self.frag_contrastive_project_head(inputs["frag_node_embedding"])
        return frag_embedding
    

