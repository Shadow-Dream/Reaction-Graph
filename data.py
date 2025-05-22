import os
import numpy as np
import pickle as pkl
import random
from tqdm import tqdm
import torch
import dgl
from itertools import groupby

class Dataset:
    def __init__(self,dataset_dir,device = "cuda"):
        self.dataset_dir = dataset_dir
        self.device = device

    def __iter__(self):
        return self
    
    def __next__(self):
        pass

class ConditionDataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 type,
                 merge,
                 device,
                 search_for_condition,
                 encode_search_result,
                 dataset_prefix,
                 shuffle,
                 dataset_filenames):
        super().__init__(dataset_dir,device=device)
        self.type = type
        self.dataset_dir = dataset_dir
        self.dataset_filename_list = os.listdir(dataset_dir)
        self.dataset_filenames = []
        dataset_prefix = f"{dataset_prefix}{type.capitalize()}"
        with open(f"{dataset_dir}/USPTO_Condition_Solvent.pkl","rb") as f:
            self.solvents = np.array(pkl.load(f))
        with open(f"{dataset_dir}/USPTO_Condition_Catalyst.pkl","rb") as f:
            self.catalysts = np.array(pkl.load(f))
        with open(f"{dataset_dir}/USPTO_Condition_Reagent.pkl","rb") as f:
            self.reagents = np.array(pkl.load(f))
        if dataset_filenames is None:
            for dataset_filename in self.dataset_filename_list:
                if dataset_filename.startswith(dataset_prefix) and dataset_filename.endswith(".pkl"):
                    self.dataset_filenames.append(f"{dataset_dir}/{dataset_filename}")
        else:
            self.dataset_filenames = dataset_filenames
        if search_for_condition:
            dataset_dictionary = f"USPTO_Condition_Reaction_Dictionary_{type.capitalize()}.pkl"
            with open(f"{self.dataset_dir}/{dataset_dictionary}","rb") as f:
                self.dataset_dictionary = pkl.load(f)
                
        self.dataset_index = 0
        self.dataset_offset = -1
        self.dataset_buffer = None
        self.merge = merge

        self.load()
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_dataset()
        self.postprocess_dataset()
        self.search_for_condition = search_for_condition
        self.encode_search_result = encode_search_result
        
        
    def __len__(self):
        return self.length

    def shuffle_dataset_filenames(self):
        random.shuffle(self.dataset_filenames)

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def postprocess_dataset(self):
        for batch in tqdm(self.dataset_buffer):

            inputs = batch["inputs"]
            fingerprints = inputs["fingerprint"]
            message_graph = inputs["message_graph"]
            reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
            product_message_passing_graph = inputs["product_message_passing_graph"]
            reactant_geometry_message_graph = inputs["reactant_geometry_message_graph"]
            product_geometry_message_graph = inputs["product_geometry_message_graph"]
            reaction_atom_message_passing_graph = inputs["reaction_atom_message_passing_graph"]

            fingerprint_tensor = torch.zeros([message_graph.num_nodes(),1024]).byte()

            for fingerprint_index,fingerprint_vector in enumerate(fingerprints):
                for fingerprint_bit in fingerprint_vector:
                    fingerprint_tensor[fingerprint_index,fingerprint_bit % 1024] = 1

            message_graph.ndata["fingerprint"] = fingerprint_tensor

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
            
            reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5]).float().to(reactant_atom_edge_attribute.device)
            reactant_atom_padding[:,0] = 1
            reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

            product_atom_padding = torch.zeros([product_atom_num_edge,5]).float().to(product_atom_edge_attribute.device)
            product_atom_padding[:,1] = 1
            product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

            reactant_geometry_atom_edge_attribute = torch.zeros([reactant_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            reactant_geometry_atom_edge_attribute[:,-1] = 1

            product_geometry_atom_edge_attribute = torch.zeros([product_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            product_geometry_atom_edge_attribute[:,-1] = 1

            reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2]).float().to(reaction_atom_edge_attribute.device)
            reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1]).float().to(reaction_atom_edge_attribute.device)
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
            reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge]).float()

            message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reactant_geometry_message_graph,product_geometry_message_graph,reaction_atom_message_passing_graph])
            
            node_type = message_graph.ndata["type"]
            select_reactant_atoms = node_type[:,0] == 1
            select_product_atoms = node_type[:,1] == 1
            select_reactant_molecule = node_type[:,2] == 1
            select_product_molecule = node_type[:,3] == 1
            select_atoms = select_reactant_atoms | select_product_atoms
            select_molecule = select_reactant_molecule | select_product_molecule

            device = node_type.device

            def get_num_node_each_batch(vector):
                count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
                count_list = torch.tensor(count_list).to(device)
                return count_list
            
            num_atoms_each_batch = get_num_node_each_batch(select_atoms)
            
            indices = torch.arange(node_type.size(0)).to(node_type.device)
            remove_indices = indices[select_molecule]
            
            message_graph.remove_nodes(remove_indices)
            message_passing_graph.remove_nodes(remove_indices)
            message_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_edges(num_edges_each_batch)

            reaction_graph = message_passing_graph
            
            reaction_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()

            batch["inputs"] = reaction_graph

    def encode_condition(self,row):
        catalyst1 = str(row[0]) == self.catalysts
        solvent1 = str(row[1]) == self.solvents
        solvent2 = str(row[2]) == self.solvents
        reagent1 = str(row[3]) == self.reagents
        reagent2 = str(row[4]) == self.reagents
        encoding = np.concatenate([catalyst1,solvent1,solvent2,reagent1,reagent2]).astype(np.float32)
        return encoding

    def process(self,batch):
        if self.search_for_condition:
            conditions = []
            for key in batch["keys"]:
                assert key in self.dataset_dictionary.keys(), f"key {key} not in dataset dictionary"
                if self.encode_search_result:
                    condition = np.array([self.encode_condition(condition_sample) for condition_sample in self.dataset_dictionary[key]])
                    condition = torch.tensor(condition).float().to(self.device)
                else:
                    condition = self.dataset_dictionary[key]
                conditions.append(condition)
        else:
            conditions = torch.tensor(batch["outputs"]).float().to(self.device)
        keys = batch["keys"]
        inputs = batch["inputs"]
        message_graph = inputs["message_graph"].to(self.device)
        message_passing_graph = inputs["message_passing_graph"].to(self.device)
        message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
        message_graph.ndata["fingerprint"] = message_graph.ndata["fingerprint"].float()
        message_graph.ndata["leaving_group"] = message_graph.ndata["leaving_group"].long()
        message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
        inputs = {"message_graph":message_graph,
                  "message_passing_graph":message_passing_graph}
        new_batch = {"inputs":inputs,"outputs":conditions,"keys":keys}
        return new_batch

    def load(self):
        self.dataset_buffer = []
        if self.merge:
            for dataset_filename in tqdm(self.dataset_filenames):
                with open(dataset_filename,"rb") as f:
                    self.dataset_buffer += pkl.load(f)
            self.length = len(self.dataset_buffer)
            
        else:
            with open(f"{self.dataset_dir}/{self.dataset_filenames[self.dataset_index]}","rb") as f:
                self.dataset_buffer = pkl.load(f)
            
    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = -1
            if self.merge:
                if self.shuffle:
                    self.shuffle_dataset()
                raise StopIteration
            else:
                self.dataset_index += 1
                if self.dataset_index == len(self.dataset_filenames):
                    self.dataset_offset = -1
                    self.dataset_index = 0
                    if self.shuffle:
                        self.shuffle_dataset_filenames()
                    self.load()
                    raise StopIteration
                self.load()
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch
    
class YieldDataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 type,
                 merge = True,
                 device = "cuda",
                 length = 0,
                 dataset_prefix = "ReactionYieldGraph",
                 multiply = 1,
                 shuffle = True,
                 dataset_filenames = None):
        super().__init__(dataset_dir,device=device)
        self.type = type
        self.multiply = multiply
        self.dataset_dir = dataset_dir
        self.dataset_filename_list = os.listdir(dataset_dir)
        self.dataset_filenames = []
        dataset_prefix = f"{dataset_prefix}{type.capitalize()}"
        if dataset_filenames is None:
            for dataset_filename in self.dataset_filename_list:
                if dataset_filename.startswith(dataset_prefix) and dataset_filename.endswith(".pkl"):
                    self.dataset_filenames.append(f"{dataset_dir}/{dataset_filename}")
        else:
            self.dataset_filenames = dataset_filenames

        self.length = length
        self.dataset_index = 0
        self.dataset_offset = -1
        self.dataset_buffer = None
        self.merge = merge
        self.load()
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_dataset()
        self.postprocess_dataset()
        
        
    def __len__(self):
        return self.length

    def shuffle_dataset_filenames(self):
        random.shuffle(self.dataset_filenames)

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def postprocess_dataset(self):
        output_list = []
        for batch in tqdm(self.dataset_buffer):
            inputs = batch["inputs"]
            outputs = batch["outputs"]
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
            
            reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5]).float().to(reactant_atom_edge_attribute.device)
            reactant_atom_padding[:,0] = 1
            reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

            product_atom_padding = torch.zeros([product_atom_num_edge,5]).float().to(product_atom_edge_attribute.device)
            product_atom_padding[:,1] = 1
            product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

            reactant_geometry_atom_edge_attribute = torch.zeros([reactant_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            reactant_geometry_atom_edge_attribute[:,-1] = 1

            product_geometry_atom_edge_attribute = torch.zeros([product_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            product_geometry_atom_edge_attribute[:,-1] = 1

            reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2]).float().to(reaction_atom_edge_attribute.device)
            reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1]).float().to(reaction_atom_edge_attribute.device)
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
            reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge]).float()

            message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reactant_geometry_message_graph,product_geometry_message_graph,reaction_atom_message_passing_graph])
            
            node_type = message_graph.ndata["type"]
            select_reactant_atoms = (node_type[:,0] == 1) | (node_type[:,4] == 1)
            select_product_atoms = node_type[:,1] == 1
            select_reactant_molecule = node_type[:,2] == 1
            select_product_molecule = node_type[:,3] == 1
            select_atoms = select_reactant_atoms | select_product_atoms
            select_molecule = select_reactant_molecule | select_product_molecule

            device = node_type.device

            def get_num_node_each_batch(vector):
                count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
                count_list = torch.tensor(count_list).to(device)
                return count_list
            
            num_atoms_each_batch = get_num_node_each_batch(select_atoms)
            
            indices = torch.arange(node_type.size(0)).to(node_type.device)
            remove_indices = indices[select_molecule]
            
            message_graph.remove_nodes(remove_indices)
            message_passing_graph.remove_nodes(remove_indices)
            message_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_edges(num_edges_each_batch)

            message_graph.ndata["attribute"] = torch.cat([message_graph.ndata["attribute"],message_graph.ndata["type"]],dim = -1)

            outputs = outputs*self.multiply

            reaction_graph  = message_passing_graph
            reaction_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()

            batch["inputs"] = reaction_graph
            batch["outputs"] = outputs
            output_list.append(outputs)
        outputs = np.concatenate(output_list)
        self.mean = np.mean(outputs)
        self.std = np.std(outputs)
        self.var = np.var(outputs)
            

    def process(self,batch):
        inputs = batch["inputs"]
        outputs = torch.tensor(batch["outputs"]).float().to(self.device)
        message_graph = inputs["message_graph"].to(self.device)
        message_passing_graph = inputs["message_passing_graph"].to(self.device)
        message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
        message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
        inputs = {"message_graph":message_graph,
                  "message_passing_graph":message_passing_graph}
        new_batch = {"inputs":inputs,"outputs":outputs}
        return new_batch

    def load(self):
        if self.merge:
            for dataset_filename in tqdm(self.dataset_filenames):
                with open(dataset_filename,"rb") as f:
                    buffer = pkl.load(f)
                    if self.dataset_buffer is None:
                        self.dataset_buffer = buffer
                    else:
                        self.dataset_buffer += buffer
            self.length = len(self.dataset_buffer)
            
        else:
            with open(f"{self.dataset_dir}/{self.dataset_filenames[self.dataset_index]}","rb") as f:
                self.dataset_buffer = pkl.load(f)
            
    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = 0
            if self.merge:
                if self.shuffle:
                    self.shuffle_dataset()
                raise StopIteration
            else:
                self.dataset_index += 1
                if self.dataset_index == len(self.dataset_filenames):
                    self.dataset_offset = -1
                    self.dataset_index = 0
                    if self.shuffle:
                        self.shuffle_dataset_filenames()
                    self.load()
                    raise StopIteration
                self.load()
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch

class TypeDataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 type,
                 merge = True,
                 device = "cuda",
                 length = 0,
                 dataset_prefix = "ReactionTypeGraph",
                 shuffle = True,
                 dataset_filenames = None):
        super().__init__(dataset_dir,device=device)
        self.type = type
        self.dataset_dir = dataset_dir
        self.dataset_filename_list = os.listdir(dataset_dir)
        self.dataset_filenames = []
        dataset_prefix = f"{dataset_prefix}{type.capitalize()}"
        if dataset_filenames is None:
            for dataset_filename in self.dataset_filename_list:
                if dataset_filename.startswith(dataset_prefix) and dataset_filename.endswith(".pkl"):
                    self.dataset_filenames.append(f"{dataset_dir}/{dataset_filename}")
        else:
            self.dataset_filenames = dataset_filenames
        self.length = length
        self.dataset_index = 0
        self.dataset_offset = -1
        self.dataset_buffer = None
        self.merge = merge
        self.load()
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_dataset()
        self.postprocess_dataset()
        
        
    def __len__(self):
        return self.length

    def shuffle_dataset_filenames(self):
        random.shuffle(self.dataset_filenames)

    def shuffle_dataset(self):
        random.shuffle(self.dataset_buffer)

    def postprocess_dataset(self):
        for batch in tqdm(self.dataset_buffer):
            inputs = batch["inputs"]
            message_graph = inputs["message_graph"]
            reactant_message_passing_graph = inputs["reactant_message_passing_graph"]
            product_message_passing_graph = inputs["product_message_passing_graph"]
            reactant_geometry_message_graph = inputs["reactant_geometry_message_graph"]
            product_geometry_message_graph = inputs["product_geometry_message_graph"]
            reaction_atom_message_passing_graph = inputs["reaction_atom_message_passing_graph"]

            message_graph.ndata["attribute"] = torch.cat([message_graph.ndata["attribute"],message_graph.ndata["type"]],dim = -1)

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
            
            reactant_atom_padding = torch.zeros([reactant_atom_num_edge,5]).float().to(reactant_atom_edge_attribute.device)
            reactant_atom_padding[:,0] = 1
            reactant_atom_edge_attribute = torch.cat([reactant_atom_edge_attribute,reactant_atom_padding],dim = -1)

            product_atom_padding = torch.zeros([product_atom_num_edge,5]).float().to(product_atom_edge_attribute.device)
            product_atom_padding[:,1] = 1
            product_atom_edge_attribute = torch.cat([product_atom_edge_attribute,product_atom_padding],dim = -1)

            reactant_geometry_atom_edge_attribute = torch.zeros([reactant_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            reactant_geometry_atom_edge_attribute[:,-1] = 1

            product_geometry_atom_edge_attribute = torch.zeros([product_geometry_atom_num_edge,dim_edge_attribute + 5]).float().to(reaction_atom_edge_attribute.device)
            product_geometry_atom_edge_attribute[:,-1] = 1

            reaction_atom_padding_left = torch.zeros([reaction_atom_num_edge,dim_edge_attribute + 2]).float().to(reaction_atom_edge_attribute.device)
            reaction_atom_padding_right = torch.zeros([reaction_atom_num_edge,1]).float().to(reaction_atom_edge_attribute.device)
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
            reaction_atom_message_passing_graph.edata['length'] = torch.zeros([reaction_atom_num_edge]).float()

            message_passing_graph = dgl.merge([reactant_message_passing_graph,product_message_passing_graph,reactant_geometry_message_graph,product_geometry_message_graph,reaction_atom_message_passing_graph])
            
            node_type = message_graph.ndata["type"]
            select_reactant_atoms = (node_type[:,0] == 1) | (node_type[:,4] == 1)
            select_product_atoms = node_type[:,1] == 1
            select_reactant_molecule = node_type[:,2] == 1
            select_product_molecule = node_type[:,3] == 1
            select_atoms = select_reactant_atoms | select_product_atoms
            select_molecule = select_reactant_molecule | select_product_molecule

            device = node_type.device

            def get_num_node_each_batch(vector):
                count_list = [sum(1 for _ in group) for key, group in groupby(vector) if key == 1]
                count_list = torch.tensor(count_list).to(device)
                return count_list
            
            num_atoms_each_batch = get_num_node_each_batch(select_atoms)
            
            indices = torch.arange(node_type.size(0)).to(node_type.device)
            remove_indices = indices[select_molecule]
            
            message_graph.remove_nodes(remove_indices)
            message_passing_graph.remove_nodes(remove_indices)
            message_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_nodes(num_atoms_each_batch)
            message_passing_graph.set_batch_num_edges(num_edges_each_batch)

            reaction_graph = message_passing_graph
            reaction_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()

            batch["inputs"] = reaction_graph

    def process(self,batch):
        inputs = batch["inputs"]
        outputs = torch.tensor(batch["outputs"]).float().to(self.device)
        message_graph = inputs["message_graph"].to(self.device)
        message_passing_graph = inputs["message_passing_graph"].to(self.device)
        message_graph.ndata["attribute"] = message_graph.ndata["attribute"].float()
        message_passing_graph.edata["attribute"] = message_passing_graph.edata["attribute"].float()
        inputs = {"message_graph":message_graph,
                  "message_passing_graph":message_passing_graph}
        new_batch = {"inputs":inputs,"outputs":outputs}
        return new_batch

    def load(self):
        if self.merge:
            for dataset_filename in tqdm(self.dataset_filenames):
                with open(dataset_filename,"rb") as f:
                    buffer = pkl.load(f)
                    if self.dataset_buffer is None:
                        self.dataset_buffer = buffer
                    else:
                        self.dataset_buffer += buffer
            self.length = len(self.dataset_buffer)
            
        else:
            with open(f"{self.dataset_dir}/{self.dataset_filenames[self.dataset_index]}","rb") as f:
                self.dataset_buffer = pkl.load(f)
            
    def __next__(self):
        self.dataset_offset += 1
        if self.dataset_offset == len(self.dataset_buffer):
            self.dataset_offset = 0
            if self.merge:
                if self.shuffle:
                    self.shuffle_dataset()
                raise StopIteration
            else:
                self.dataset_index += 1
                if self.dataset_index == len(self.dataset_filenames):
                    self.dataset_offset = -1
                    self.dataset_index = 0
                    if self.shuffle:
                        self.shuffle_dataset_filenames()
                    self.load()
                    raise StopIteration
                self.load()
        batch = self.dataset_buffer[self.dataset_offset]
        batch = self.process(batch)
        return batch

