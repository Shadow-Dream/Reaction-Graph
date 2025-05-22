import numpy as np
import os
from tqdm import tqdm
import torch
import pickle as pkl
for file in os.listdir("graph"):
    coordinate_file = file.replace(".pkl",".npy")
    with open("graph/"+file,"rb") as f:
        batches = pkl.load(f)
    coordinates = np.load("coordinates/"+coordinate_file)
    for batch in tqdm(batches):
        inputs = batch["inputs"]
        message_graph = inputs["message_graph"]
        reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
        product_message_passing_graph = inputs["product_message_passing_graph"]
        reactant_geometry_message_graph = inputs["reactant_geometry_message_graph"]
        product_geometry_message_graph = inputs["product_geometry_message_graph"]

        type = message_graph.ndata["type"]
        select_atoms = (type[:,0]==1) | (type[:,1]==1)
        num_nodes = select_atoms.sum()
        message_graph.ndata["position"][select_atoms] = torch.tensor(coordinates[:num_nodes],dtype = torch.float32)
        coordinates = coordinates[num_nodes:]
        positions = message_graph.ndata["position"]
        def get_direction(graph):
            src = graph.edges()[0]
            dst = graph.edges()[1]
            directions = positions[dst] - positions[src]
            graph.edata["direction"] = directions

        get_direction(reactant_message_passing_graph)
        get_direction(product_message_passing_graph)
        get_direction(reactant_geometry_message_graph)
        get_direction(product_geometry_message_graph)

        inputs["message_graph"] = message_graph
        inputs["reactant_message_passing_graph"] = reactant_message_passing_graph
        inputs["product_message_passing_graph"] = product_message_passing_graph
        inputs["reactant_geometry_message_graph"] = reactant_geometry_message_graph
        inputs["product_geometry_message_graph"] = product_geometry_message_graph

        batch["inputs"] = inputs

    with open("graph/"+file,"wb") as f:
        pkl.dump(batches,f)
        