import pandas as pd
import pickle as pkl
import os
import dgl
import torch
from itertools import groupby
from tqdm import tqdm

files = os.listdir("graph")
for file in files:
    with open("graph/"+file,"rb") as f:
        batches = pkl.load(f)
        for batch in tqdm(batches):
            inputs = batch["inputs"]
            message_graph = inputs["message_graph"]
            reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
            product_message_passing_graph = inputs["product_message_passing_graph"]
            reactant_geometry_message_graph = inputs["reactant_geometry_message_graph"]
            product_geometry_message_graph = inputs["product_geometry_message_graph"]
            reaction_atom_message_passing_graph = inputs["reaction_atom_message_passing_graph"]

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
            reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge])

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
            batch["inputs"] = inputs
    file = file.replace("_9600","")
    with open("reaction_graph/"+file,"wb") as f:
        pkl.dump(batches,f)