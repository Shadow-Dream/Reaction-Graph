import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as func
from dgl.nn.pytorch import NNConv, Set2Set
from torch.optim import Adam
from torch.nn import CrossEntropyLoss,MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
from tqdm import tqdm
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from pycm import ConfusionMatrix
import data

class RBFEmbedding(Module):
    def __init__(self, dim,**kwargs):
        super(RBFEmbedding, self).__init__()
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.dim = dim
        self.mean = torch.nn.Parameter(torch.Tensor(dim))
        self.std = torch.nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.mean, 0, 1)
        torch.nn.init.constant_(self.std, 10)

    def forward(self, lengths):
        lengths = lengths.view(-1, 1)
        mean = self.mean.view(1, -1)
        centralized_lengths = lengths - mean
        squared_lengths = torch.square(centralized_lengths)
        length_embedding = torch.exp(-self.std * squared_lengths)
        return length_embedding

class RBFMPNN(Module):
    def __init__(self,
                 dim_node_attribute,
                 dim_edge_attribute,
                 dim_edge_length,
                 dim_hidden_features,
                 dim_hidden,
                 message_passing_step,
                 pooling_step,
                 num_layers_pooling,
                 **kwargs):
        
        super(RBFMPNN, self).__init__()
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.node_embedding = nn.Sequential(nn.Linear(dim_node_attribute, dim_hidden_features), nn.ReLU())
        self.edge_embedding = RBFEmbedding(dim_edge_length)
        self.bond_function = nn.Linear(dim_edge_length + dim_edge_attribute, dim_hidden_features * dim_hidden_features)
        self.gnn = NNConv(dim_hidden_features, dim_hidden_features, self.bond_function, 'sum')
        self.gru = nn.GRU(dim_hidden_features, dim_hidden_features)
        self.pooling = Set2Set(input_dim = dim_hidden_features * 2,
                               n_iters = pooling_step,
                               n_layers = num_layers_pooling)

        self.sparsify = nn.Sequential(nn.Linear(dim_hidden_features * 4, dim_hidden), nn.PReLU())
        self.activation = nn.ReLU()
        self.message_passing_step = message_passing_step

    def forward(self, reaction_graph):
        node_attribute = reaction_graph.ndata["attribute"]
        node_features = self.node_embedding(node_attribute)
        
        edge_attribute = reaction_graph.edata["attribute"]
        edge_length = self.edge_embedding(reaction_graph.edata["length"])
        edge_features = torch.cat([edge_attribute,edge_length],dim = -1)
        
        node_hiddens = node_features.unsqueeze(0)
        node_aggregation = node_features

        for _ in range(self.message_passing_step):
            node_features = self.gnn(reaction_graph,node_features,edge_features)
            node_features = self.activation(node_features).unsqueeze(0)
            node_features, node_hiddens = self.gru(node_features, node_hiddens)
            node_features = node_features.squeeze(0)

        node_aggregation = torch.cat([node_features,node_aggregation],dim = -1)
        reaction_features = self.pooling(reaction_graph, node_aggregation)
        reaction_features = self.sparsify(reaction_features)
        return reaction_features

# TODO: Add support for more types of reaction conditions, rather than being limited to five fixed outputs.
class ConditionOutputHead(Module):
    def __init__(self, 
                 dim_catalyst = 54,
                 dim_solvent = 87,
                 dim_reagent = 235,
                 dim_hidden = 512,
                 dim_message = 256,
                 **kwargs):
        super(ConditionOutputHead, self).__init__()
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.inputs = nn.ModuleList([
            nn.Sequential(nn.Linear(dim_catalyst,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_solvent,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_solvent,dim_message),nn.ReLU()),
            nn.Sequential(nn.Linear(dim_reagent,dim_message),nn.ReLU())])
        
        self.hiddens = nn.ModuleList([
            nn.Sequential(nn.Linear(dim_hidden + i * dim_message,dim_hidden),
                          nn.ReLU(),
                          nn.Linear(dim_hidden, dim_hidden),
                          nn.Tanh()) for i in range(5)])
        self.outputs = nn.ModuleList([
            nn.Linear(dim_hidden,dim_catalyst),
            nn.Linear(dim_hidden,dim_solvent),
            nn.Linear(dim_hidden,dim_solvent),
            nn.Linear(dim_hidden,dim_reagent),
            nn.Linear(dim_hidden,dim_reagent),
        ])
        self.dim_catalyst = dim_catalyst
        self.dim_solvent = dim_solvent
        self.dim_reagent = dim_reagent
    
    def catalyst1(self,reaction_features):
        return self.outputs[0](self.hiddens[0](reaction_features))
    
    def solvent1(self,reaction_features,catalyst1):
        catalyst1 = self.inputs[0](catalyst1.float())
        reaction_features = torch.cat([reaction_features,catalyst1],dim = -1)
        return self.outputs[1](self.hiddens[1](reaction_features))
    
    def solvent2(self,reaction_features,catalyst1,solvent1):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        reaction_features = torch.cat([reaction_features,catalyst1,solvent1],dim = -1)
        return self.outputs[2](self.hiddens[2](reaction_features))
    
    def reagent1(self,reaction_features,catalyst1,solvent1,solvent2):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        solvent2 = self.inputs[2](solvent2.float())
        reaction_features = torch.cat([reaction_features,catalyst1,solvent1,solvent2],dim = -1)
        return self.outputs[3](self.hiddens[3](reaction_features))
    
    def reagent2(self,reaction_features,catalyst1,solvent1,solvent2,reagent1):
        catalyst1 = self.inputs[0](catalyst1.float())
        solvent1 = self.inputs[1](solvent1.float())
        solvent2 = self.inputs[2](solvent2.float())
        reagent1 = self.inputs[3](reagent1.float())
        reaction_features = torch.cat([reaction_features,catalyst1,solvent1,solvent2,reagent1],dim = -1)
        return self.outputs[4](self.hiddens[4](reaction_features))
    
    @torch.no_grad()
    def inference(self,reaction_features,beams = [1,3,1,5,1]):
        batch_size = reaction_features.shape[0]

        catalyst1 = func.softmax(self.catalyst1(reaction_features),dim=-1)
        catalyst1_score,catalyst1_topk = torch.topk(catalyst1,beams[0],dim=-1)
        catalyst1_topk = func.one_hot(catalyst1_topk,self.dim_catalyst)

        solvent1_scores = []
        solvent1_topks = []
        for index in range(beams[0]):
            solvent1 = func.softmax(self.solvent1(reaction_features,catalyst1_topk[:,index]),dim=-1)
            solvent1_score,solvent1_topk = torch.topk(solvent1,beams[1],dim=-1)
            solvent1_topk = func.one_hot(solvent1_topk,self.dim_solvent)
            solvent1_scores.append(solvent1_score)
            solvent1_topks.append(solvent1_topk)
        solvent1_score = torch.concat(solvent1_scores,1)
        solvent1_topk = torch.concat(solvent1_topks,1)

        solvent2_scores = []
        solvent2_topks = []
        for index in range(beams[0]*beams[1]):
            solvent2 = func.softmax(self.solvent2(reaction_features,catalyst1_topk[:,index//beams[1]],solvent1_topk[:,index]),dim=-1)
            solvent2_score,solvent2_topk = torch.topk(solvent2,beams[2],dim=-1)
            solvent2_topk = func.one_hot(solvent2_topk,self.dim_solvent)
            solvent2_scores.append(solvent2_score)
            solvent2_topks.append(solvent2_topk)
        solvent2_score = torch.concat(solvent2_scores,1)
        solvent2_topk = torch.concat(solvent2_topks,1)

        reagent1_scores = []
        reagent1_topks = []
        for index in range(beams[0]*beams[1]*beams[2]):
            reagent1 = func.softmax(self.reagent1(reaction_features,catalyst1_topk[:,index//(beams[1]*beams[2])],solvent1_topk[:,index//beams[2]],solvent2_topk[:,index]),dim=-1)
            reagent1_score,reagent1_topk = torch.topk(reagent1,beams[3],dim=-1)
            reagent1_topk = func.one_hot(reagent1_topk,self.dim_reagent)
            reagent1_scores.append(reagent1_score)
            reagent1_topks.append(reagent1_topk)
        reagent1_score = torch.concat(reagent1_scores,1)
        reagent1_topk = torch.concat(reagent1_topks,1)

        reagent2_scores = []
        reagent2_topks = []
        for index in range(beams[0]*beams[1]*beams[2]*beams[3]):
            reagent2 = func.softmax(self.reagent2(reaction_features,catalyst1_topk[:,index//(beams[1]*beams[2]*beams[3])],solvent1_topk[:,index//(beams[2]*beams[3])],solvent2_topk[:,index//beams[3]],reagent1_topk[:,index]),dim=-1)
            reagent2_score,reagent2_topk = torch.topk(reagent2,beams[4],dim=-1)
            reagent2_topk = func.one_hot(reagent2_topk,self.dim_reagent)
            reagent2_scores.append(reagent2_score)
            reagent2_topks.append(reagent2_topk)
        reagent2_score = torch.concat(reagent2_scores,1)
        reagent2_topk = torch.concat(reagent2_topks,1)

        catalyst1_topk = catalyst1_topk.repeat(1,1,beams[1]*beams[2]*beams[3]*beams[4]).reshape(batch_size,-1,self.dim_catalyst)
        solvent1_topk = solvent1_topk.repeat(1,1,beams[2]*beams[3]*beams[4]).reshape(batch_size,-1,self.dim_solvent)
        solvent2_topk = solvent2_topk.repeat(1,1,beams[3]*beams[4]).reshape(batch_size,-1,self.dim_solvent)
        reagent1_topk = reagent1_topk.repeat(1,1,beams[4]).reshape(batch_size,-1,self.dim_reagent)

        catalyst1_score = catalyst1_score.unsqueeze(-1).repeat(1,1,beams[1]*beams[2]*beams[3]*beams[4]).reshape(batch_size,-1)
        solvent1_score = solvent1_score.unsqueeze(-1).repeat(1,1,beams[2]*beams[3]*beams[4]).reshape(batch_size,-1)
        solvent2_score = solvent2_score.unsqueeze(-1).repeat(1,1,beams[3]*beams[4]).reshape(batch_size,-1)
        reagent1_score = reagent1_score.unsqueeze(-1).repeat(1,1,beams[4]).reshape(batch_size,-1)

        _,catalyst1_topk = catalyst1_topk.max(-1)
        _,solvent1_topk = solvent1_topk.max(-1)
        _,solvent2_topk = solvent2_topk.max(-1)
        _,reagent1_topk = reagent1_topk.max(-1)
        _,reagent2_topk = reagent2_topk.max(-1)

        catalyst1_topk = catalyst1_topk.unsqueeze(-1)
        solvent1_topk = solvent1_topk.unsqueeze(-1)
        solvent2_topk = solvent2_topk.unsqueeze(-1)
        reagent1_topk = reagent1_topk.unsqueeze(-1)
        reagent2_topk = reagent2_topk.unsqueeze(-1)

        topk = torch.cat([catalyst1_topk,solvent1_topk,solvent2_topk,reagent1_topk,reagent2_topk],-1)
        score = catalyst1_score * solvent1_score * solvent2_score * reagent1_score * reagent2_score
        _,sort_indices= score.sort(-1,descending=True)
        batch_size,beam_sum = sort_indices.shape
        sort_indices = sort_indices + torch.arange(0,batch_size).to(sort_indices.device).unsqueeze(-1) * beam_sum
        sort_indices = sort_indices.view(-1)
        topk = topk.view(-1,5)[sort_indices].view(batch_size,-1,5)
        return topk
    
    def forward(**kwargs):
        raise NotImplementedError("Please use catalyst1, solvent1, solvent2, reagent1, reagent2, or inference method.")
    
class ConditionModel:
    def __init__(self,
                 dataset_train,
                 dataset_test,
                 dataset_val,
                 model_dir,
                 device,
                 dim_catalyst,
                 dim_solvent,
                 dim_reagent,
                 parameters_for_model,
                 parameters_for_optimizer_stage_one,
                 parameters_for_scheduler_stage_one,
                 parameters_for_optimizer_stage_two,
                 parameters_for_scheduler_stage_two,
                 none_weights,
                 smoothing,
                 accumulation_steps,
                 max_gradient,
                 **kwargs):
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        
        self.device = device
        self.model_dir = model_dir

        self.smoothing = smoothing
        self.accmulation_steps = accumulation_steps
        self.max_gradient = max_gradient
        self.model = RBFMPNN(**parameters_for_model).to(device)
        self.output = ConditionOutputHead(dim_catalyst,dim_solvent,dim_reagent,**parameters_for_model).to(device)

        self.cross_entropy_loss = CrossEntropyLoss(reduce="none")

        stage_one_parameters = []
        stage_one_parameters += list(self.model.parameters())
        stage_one_parameters += list(self.output.parameters())
        self.optimizer_stage_one = Adam(stage_one_parameters,**parameters_for_optimizer_stage_one)
        self.scheduler_stage_one = ReduceLROnPlateau(self.optimizer_stage_one,**parameters_for_scheduler_stage_one)

        stage_two_parameters = []
        stage_two_parameters += list(self.output.inputs.parameters())
        stage_two_parameters += list(self.output.hiddens.parameters())
        stage_two_parameters += list(self.output.outputs.parameters())
        stage_two_parameters += list(self.model.sparsify.parameters())
        self.optimizer_stage_two = Adam(stage_two_parameters,**parameters_for_optimizer_stage_two)
        self.scheduler_stage_two = ReduceLROnPlateau(self.optimizer_stage_two,**parameters_for_scheduler_stage_two)

        self.split_lengths = [0,dim_catalyst,dim_solvent,dim_solvent,dim_reagent,dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]
        self.init_none_weights(none_weights)

    def init_none_weights(self,none_weights):
        self.learning_rates = {}
        dataset = self.dataset_test if self.dataset_test is not None else self.dataset_val if self.dataset_val is not None else self.dataset_train
        if dataset is None:
            print("Warning: No dataset is provided, can not initialize none weights.")
            return
        self.learning_rates["catalyst1"] = torch.ones([len(dataset.catalysts)]).to(self.device)
        self.learning_rates["catalyst1"][int((dataset.catalysts == "nan").argmax())] = none_weights["catalyst1"]
        self.learning_rates["solvent1"] = torch.ones([len(dataset.solvents)]).to(self.device)
        self.learning_rates["solvent1"][int((dataset.solvents == "nan").argmax())] = none_weights["solvent1"]
        self.learning_rates["solvent2"] = torch.ones([len(dataset.solvents)]).to(self.device)
        self.learning_rates["solvent2"][int((dataset.solvents == "nan").argmax())] = none_weights["solvent2"]
        self.learning_rates["reagent1"] = torch.ones([len(dataset.reagents)]).to(self.device)
        self.learning_rates["reagent1"][int((dataset.reagents == "nan").argmax())] = none_weights["reagent1"]
        self.learning_rates["reagent2"] = torch.ones([len(dataset.reagents)]).to(self.device)
        self.learning_rates["reagent2"][int((dataset.reagents == "nan").argmax())] = none_weights["reagent2"]

    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient)

    def load(self,filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict,strict=False)
        self.output.load_state_dict(state_dict,strict=False)

    def save(self,postfix):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        state_dict = {}
        state_dict.update(self.model.state_dict())
        state_dict.update(self.output.state_dict())
        torch.save(state_dict,f"{self.model_dir}/{model_name}")

    def get_category_learning_rate(self,real,category):
        return (self.learning_rates[category]*real).sum(-1)
    
    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.sparsify.parameters():
            param.requires_grad = True

    def reset_parameters(self):
        for param in self.output.inputs.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.output.hiddens.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.output.outputs.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.output.reaction_nn.sparsify.parameters():
            nn.init.normal_(param,std=0.1)

    def train(self,epoches_stage_one,epoches_stage_two,save_delta,progress_bar):
        for epoch in range(epoches_stage_one + epoches_stage_two):
            if epoch == epoches_stage_one:
                self.freeze_parameters()
                self.reset_parameters()
            
            self.model.train()
            self.output.train()
            progress = tqdm(enumerate(self.dataset_train),total=len(self.dataset_train)) if progress_bar else enumerate(self.dataset_train)
            losses = []
            for index,batch in progress:
                inputs = batch["inputs"]
                outputs = batch["outputs"]
                real_catalyst1,real_solvent1,real_solvent2,real_reagent1,real_reagent2 = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
                if epoch < epoches_stage_one:
                    learning_rate_catalyst1 = self.get_category_learning_rate(real_catalyst1,"catalyst1")
                    learning_rate_solvent1 = self.get_category_learning_rate(real_solvent1,"solvent1")
                    learning_rate_solvent2 = self.get_category_learning_rate(real_solvent2,"solvent2")
                    learning_rate_reagent1 = self.get_category_learning_rate(real_reagent1,"reagent1")
                    learning_rate_reagent2 = self.get_category_learning_rate(real_reagent2,"reagent2")

                    smooth_real_catalyst1 = real_catalyst1 * self.smoothing[0] + torch.ones_like(real_catalyst1) * (1 - self.smoothing[0]) / real_catalyst1.shape[-1]
                    smooth_real_solvent1 = real_solvent1 * self.smoothing[1] + torch.ones_like(real_solvent1) * (1 - self.smoothing[1]) / real_solvent1.shape[-1]
                    smooth_real_solvent2 = real_solvent2 * self.smoothing[2] + torch.ones_like(real_solvent2) * (1 - self.smoothing[2]) / real_solvent2.shape[-1]
                    smooth_real_reagent1 = real_reagent1 * self.smoothing[3] + torch.ones_like(real_reagent1) * (1 - self.smoothing[3]) / real_reagent1.shape[-1]
                    smooth_real_reagent2 = real_reagent2 * self.smoothing[4] + torch.ones_like(real_reagent2) * (1 - self.smoothing[4]) / real_reagent2.shape[-1]
                    
                    reaction_features = self.model(inputs)
                    loss_catalyst1 = self.cross_entropy_loss(self.output.catalyst1(reaction_features),smooth_real_catalyst1) * learning_rate_catalyst1
                    loss_solvent1 = self.cross_entropy_loss(self.output.solvent1(reaction_features,real_catalyst1),smooth_real_solvent1) * learning_rate_solvent1
                    loss_solvent2 = self.cross_entropy_loss(self.output.solvent2(reaction_features,real_catalyst1,real_solvent1),smooth_real_solvent2) * learning_rate_solvent2
                    loss_reagent1 = self.cross_entropy_loss(self.output.reagent1(reaction_features,real_catalyst1,real_solvent1,real_solvent2),smooth_real_reagent1) * learning_rate_reagent1
                    loss_reagent2 = self.cross_entropy_loss(self.output.reagent2(reaction_features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),smooth_real_reagent2) * learning_rate_reagent2
                    loss = 0
                    loss += loss_catalyst1.mean() 
                    loss += loss_solvent1.mean() 
                    loss += loss_solvent2.mean() 
                    loss += loss_reagent1.mean() 
                    loss += loss_reagent2.mean()
                    loss /= self.accmulation_steps
                    loss.backward()
                    if (index + 1) % self.accmulation_steps == 0:
                        parameters = []
                        parameters += list(self.model.parameters())
                        parameters += list(self.output.parameters())
                        self.postprocess_gradient(parameters)
                        self.optimizer_stage_one.step()
                        self.optimizer_stage_one.zero_grad()
                else:
                    reaction_features = self.model(inputs)
                    loss_catalyst1 = self.cross_entropy_loss(self.output.catalyst1(reaction_features),real_catalyst1)
                    loss_solvent1 = self.cross_entropy_loss(self.output.solvent1(reaction_features,real_catalyst1),real_solvent1)
                    loss_solvent2 = self.cross_entropy_loss(self.output.solvent2(reaction_features,real_catalyst1,real_solvent1),real_solvent2)
                    loss_reagent1 = self.cross_entropy_loss(self.output.reagent1(reaction_features,real_catalyst1,real_solvent1,real_solvent2),real_reagent1)
                    loss_reagent2 = self.cross_entropy_loss(self.output.reagent2(reaction_features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),real_reagent2)
                    loss = 0
                    loss += loss_catalyst1.mean() 
                    loss += loss_solvent1.mean() 
                    loss += loss_solvent2.mean() 
                    loss += loss_reagent1.mean() 
                    loss += loss_reagent2.mean()
                    loss /= self.accmulation_steps
                    loss.backward()
                    if (index + 1) % self.accmulation_steps==0:
                        parameters = []
                        parameters += list(self.model.parameters())
                        parameters += list(self.output.parameters())
                        self.postprocess_gradient(parameters)
                        self.optimizer_stage_two.step()
                        self.optimizer_stage_two.zero_grad()
                
                loss = loss.detach().cpu().item()
                losses.append(loss * self.accmulation_steps)
                losses = losses[-100:]
                average_loss = torch.tensor(losses).mean().item()
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    print(f"epoch:{epoch},loss:{average_loss}")

            accuracy = self.validate(progress_bar)
            if epoch < epoches_stage_one:
                self.scheduler_stage_one.step(1 - accuracy["overall"][1])
                learning_rate = self.optimizer_stage_one.param_groups[0]['lr']
                print(f"epoch: {epoch}, learning rate: {learning_rate}")
            else:
                self.scheduler_stage_two.step(1 - accuracy["overall"][1])
                learning_rate = self.optimizer_stage_two.param_groups[0]['lr']
                print(f"epoch: {epoch}, learning rate: {learning_rate}")
            
            topk_row="topk        "
            for topk in accuracy["overall"]:
                topk_row += " "  +(str(topk) + "      ")[:6]
            print(topk_row)
            for key in accuracy:
                accuracy_row = (key + "            ")[:12]
                for topk in accuracy[key]:
                    accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
                print(accuracy_row)
            if epoch < epoches_stage_one:
                self.save_log(accuracy,self.optimizer_stage_one.param_groups[0]['lr'],f"-{epoch}")
            else:
                self.save_log(accuracy,self.optimizer_stage_two.param_groups[0]['lr'],f"-{epoch}")
            if epoch % save_delta == save_delta - 1:
                self.save(f"-{epoch}")
                self.save(f"-last")

    def save_log(self,accuracy,learning_rate,postfix):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        with open(f"{self.model_dir}/{model_name}","w") as f:
            topk_row="topk        "
            for topk in accuracy["overall"]:
                topk_row += " "  +(str(topk) + "      ")[:6]
            f.write(topk_row + "\n")
            for key in accuracy:
                accuracy_row = (key + "            ")[:12]
                for topk in accuracy[key]:
                    accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
                f.write(accuracy_row + "\n")
            f.write(f"learning rate: {learning_rate}\n")

    @torch.no_grad()
    def validate(self,progress_bar,type,metric,beams):
        
        dataset = self.dataset_val if type=="val" else self.dataset_test if type=="test" else None
        if dataset is None:
            print("Warning: No dataset is provided, can not validate.")
            return
        self.model.eval()
        self.output.eval()
        keys = ["catalyst1","solvent1","solvent2","reagent1","reagent2"]
        accuracies = {key:{topk:[] for topk in metric} for key in keys + ["overall"]}
        progress = tqdm(dataset,total=len(dataset)) if progress_bar else dataset
        for batch in progress:
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            fake = self.output.inference(self.model(inputs),beams)
            outputs = [[output[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)] for output in outputs]
            outputs = [torch.cat([torch.argmax(output[i],-1) for i in range(5)],dim=-1).reshape(-1,5) for output in outputs]
            real = outputs
            for real_item,fake_item in zip(real,fake):
                error = abs(fake_item.unsqueeze(1) - real_item)
                for topk in metric:
                    error_overall_topk = error[:topk].reshape(-1,5).sum(-1).min(0).values
                    errors_category_topk = error[:topk].reshape(-1,5).min(0).values
                    accuracy_overall = (error_overall_topk <= 1e-7)
                    accuracies_categroy = (errors_category_topk <= 1e-7)
                    accuracies["overall"][topk].append(accuracy_overall.cpu().item())
                    for index,key in enumerate(keys):
                        accuracies[key][topk].append(accuracies_categroy[index].cpu().item())

        for key in accuracies:
            for topk in metric:
                accuracies[key][topk] = torch.tensor(accuracies[key][topk]).mean()
        return accuracies
    
class YieldOutputHead(Module):
    def __init__(self,
                 dim_hidden,
                 dim_hidden_regression,
                 dropout,
                 **kwargs):
        
        super(YieldOutputHead, self).__init__()
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        
        self.regression = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, dim_hidden_regression), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden_regression, 2)
        )
        
    def forward(self, reaction_features):
        outputs = self.regression(reaction_features)
        mean = outputs[:,0]
        logvar = outputs[:,1]
        return mean, logvar
    
class YieldModel:
    def __init__(self,
                 dataset_train,
                 dataset_test,
                 dataset_val,
                 model_dir,
                 device,
                 parameters_for_model,
                 parameters_for_optimizer,
                 parameters_for_scheduler,
                 accumulation_steps,
                 max_gradient,
                 alpha,
                 beta,
                 scale,
                 **kwargs):
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        
        self.device = device
        self.model_dir = model_dir

        self.accmulation_steps = accumulation_steps
        self.max_gradient = max_gradient
        self.model = RBFMPNN(**parameters_for_model).to(device)
        self.output = YieldOutputHead(**parameters_for_model).to(device)

        self.mse_loss = MSELoss(reduce="none")

        parameters = []
        parameters += list(self.model.parameters())
        parameters += list(self.output.parameters())
        self.optimizer = Adam(parameters,**parameters_for_optimizer)
        self.scheduler = MultiStepLR(self.optimizer,**parameters_for_scheduler)

        self.alpha = alpha
        self.beta = beta
        self.scale = scale
    
    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient)

    def train(self, epoches, num_inference_pass,progress_bar):
        for epoch in range(epoches):
            self.model.train()
            self.output.train()
            losses = []
            progress = tqdm(enumerate(self.dataset_train),total=len(self.dataset_train)) if progress_bar else enumerate(self.dataset_train)
            for index,batch in progress:
                inputs = batch["inputs"]
                real_mean = batch["outputs"]
                real_mean = (real_mean - self.dataset_train.mean) / self.dataset_train.std
                
                pred_mean,pred_var = self.output(self.model(inputs))

                loss = self.mse_loss(pred_mean, real_mean)
                loss = self.alpha * loss.mean() + (1 - self.alpha) * (loss * torch.exp(-pred_var) + pred_var).mean()
                regulation = 0
                for param in self.model.parameters():
                    regulation += torch.norm(param)
                for param in self.output.parameters():
                    regulation += torch.norm(param)
                loss += self.beta * regulation
                loss /= self.accmulation_steps
                loss.backward()
                if (index + 1) % self.accmulation_steps == 0:
                    parameters = []
                    parameters += list(self.model.parameters())
                    parameters += list(self.output.parameters())
                    self.postprocess_gradient(parameters)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                loss = loss.detach().item()
                losses.append(loss*self.accmulation_steps)
            losses = torch.tensor(losses).mean().item()
            learning_rate = self.optimizer.param_groups[0]['lr']

            self.scheduler.step()
            mae, rmse, r2,aleatoric_likelihood,overall_likelihood,aleatoric_log_var,overall_log_var = self.validate(progress_bar=progress_bar,num_inference_pass=num_inference_pass)
            self.save_log(f"-{epoch}",epoch,learning_rate,mae,rmse,r2,aleatoric_likelihood,overall_likelihood,aleatoric_log_var,overall_log_var)
            print(f"Epoch {epoch} Loss: {losses} LR: {learning_rate} MAE: {mae} RMSE: {rmse} R2: {r2}")
            print(f"Aleatoric Likelihood: {aleatoric_likelihood}")
            print(f"Overall Likelihood: {overall_likelihood}")
            print(f"Aleatoric Log Var: {aleatoric_log_var}")
            print(f"Overall Log Var: {overall_log_var}")
    
    def load(self,filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict,strict=False)
        self.output.load_state_dict(state_dict,strict=False)

    def save(self,postfix):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        state_dict = {}
        state_dict.update(self.model.state_dict())
        state_dict.update(self.output.state_dict())
        torch.save(state_dict,f"{self.model_dir}/{model_name}")

    def save_log(self,postfix,epoch,lr, mae, rmse, r2,aleatoric_likelihood,overall_likelihood,aleatoric_log_var,overall_log_var):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        with open(f"{self.model_dir}/{model_name}","w") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"R2: {r2}\n")
            f.write(f"Aleatoric Likelihood: {aleatoric_likelihood}\n")
            f.write(f"Overall Likelihood: {overall_likelihood}\n")
            f.write(f"Aleatoric Log Var: {aleatoric_log_var}\n")
            f.write(f"Overall Log Var: {overall_log_var}\n")
    
    @torch.no_grad()
    def validate(self,type,num_inference_pass,progress_bar):  
        dataset = self.dataset_val if type=="val" else self.dataset_test if type=="test" else None
        if dataset is None:
            print("Warning: No dataset is provided, can not validate.")
            return      
        self.model.eval()
        self.output.eval()
        for module in self.output.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
        
        all_mean = []
        all_var = []
        progress = tqdm(dataset,total=len(dataset)) if progress_bar else dataset

        for batch in progress:
            mean_list = []
            var_list = []
            inputs = batch["inputs"]
            
            for _ in range(num_inference_pass):
                pred_mean, pred_logvar = self.output(self.model(inputs))
                mean_list.append(pred_mean.cpu().numpy())
                var_list.append(np.exp(pred_logvar.cpu().numpy()))

            all_mean.append(np.array(mean_list).transpose())
            all_var.append(np.array(var_list).transpose())

        all_mean = np.vstack(all_mean) * self.dataset_train.std + self.dataset_train.mean
        all_var = np.vstack(all_var) * self.dataset_train.std ** 2
        
        prediction = np.mean(all_mean, 1) * self.scale
        epistemic = np.var(all_mean, 1) * self.scale**2
        aleatoric = np.mean(all_var, 1) * self.scale**2
        variance = epistemic + aleatoric

        real_mean = dataset.outputs * self.scale
        mae = mean_absolute_error(real_mean, pred_mean)
        rmse = mean_squared_error(real_mean, pred_mean) ** 0.5
        r2 = r2_score(real_mean, pred_mean)

        aleatoric_likelihood = np.mean((real_mean - prediction) ** 2 / aleatoric)
        overall_likelihood = np.mean((real_mean - prediction) ** 2 / variance)
        aleatoric_log_var = np.mean(np.log(aleatoric))
        overall_log_var = np.mean(np.log(variance))
        
        return mae, rmse, r2,aleatoric_likelihood,overall_likelihood,aleatoric_log_var,overall_log_var
    
class TypeOutputHead(Module):
    def __init__(self,
                 dim_hidden,
                 dim_hidden_classification,
                 **kwargs):
        
        super(TypeModel, self).__init__()
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.classification = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden_classification), nn.PReLU(),
            nn.Linear(dim_hidden_classification, dim_hidden_classification), nn.PReLU(),
            nn.Linear(dim_hidden_classification, 1)
        )
        
    def forward(self, reaction_features):
        outputs = self.classification(reaction_features)
        return outputs
    
class TypeModel:
    def __init__(self,
                 dataset_train,
                 dataset_val,
                 dataset_test,
                 model_dir,
                 device,
                 max_gradient,
                 parameters_for_model,
                 parameters_for_optimizer,
                 parameters_for_scheduler,
                 accumulation_steps,
                 **kwargs):
        if len(kwargs) > 0: print("Warning: Unexpected Extra Args:",kwargs)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.max_gradient = max_gradient
        self.device = device
        self.model_dir = model_dir
        
        self.accmulation_steps = accumulation_steps
        
        self.model = RBFMPNN(**parameters_for_model).to(device)
        self.output = TypeOutputHead(**parameters_for_model).to(device)

        self.criterion = CrossEntropyLoss(reduce="mean")

        self.optimizer = Adam(self.model.parameters(),**parameters_for_optimizer)
        self.scheduler = ReduceLROnPlateau(self.optimizer,**parameters_for_scheduler)

    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient)

    def load(self,filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict,strict=False)
        self.output.load_state_dict(state_dict,strict=False)

    def save(self,postfix):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        state_dict = {}
        state_dict.update(self.model.state_dict())
        state_dict.update(self.output.state_dict())
        torch.save(state_dict,f"{self.model_dir}/{model_name}")

    def train(self,epoches,progress_bar,save_delta):
        for epoch in range(epoches):
            self.model.train()
            self.output.train()
            progress = tqdm(enumerate(self.dataset_train),total=len(self.dataset_train)) if progress_bar else enumerate(self.dataset_train)
            losses = []
            for index,batch in progress:
                inputs = batch["inputs"]
                real = batch["outputs"]
                fake = self.model(inputs)
                loss = self.criterion(fake,real)
                loss /= self.accmulation_steps
                loss.backward()
                if (index + 1) % self.accmulation_steps==0:
                    self.postprocess_gradient(self.model.parameters())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                loss = float(loss.detach().cpu())
                losses.append(loss)
                losses = losses[-100:]
                average_loss = float(np.array(losses).mean())
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    print(f"epoch:{epoch},loss:{average_loss}")
            
            acc,cen,mcc = self.validate(progress_bar)
            learning_rate = self.optimizer.param_groups[0]['lr']
            print(f"epoch: {epoch}, learning rate: {learning_rate}, acc: {acc}, cen: {cen}, mcc:{mcc}")
            self.scheduler.step(1-acc)
            self.save_log(epoch,self.optimizer.param_groups[0]['lr'],f"-{epoch}",acc,cen,mcc)
            if (epoch + 1) % save_delta == 0:
                self.save(f"-{epoch}-{acc}")
                self.save(f"-last")
    
    def save_log(self,epoch,lr,postfix,acc,cen,mcc):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        with open(f"{self.model_dir}/{model_name}","w") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"ACC: {acc}\n")
            f.write(f"CEN: {cen}\n")
            f.write(f"MCC: {mcc}\n")
    
    @torch.no_grad()
    def validate(self, progress_bar=True, type="val", onlyacc = True):
        dataset = self.dataset_val if type == "val" else self.dataset_test if type == "test" else None
        progress = tqdm(dataset, total=len(dataset)) if progress_bar else dataset
        
        all_real = []
        all_fake = []
        
        for batch in progress:
            inputs = batch["inputs"]
            real = batch["outputs"].argmax(-1)
            fake = self.model(inputs).argmax(-1)
            
            all_real.append(real.cpu().numpy())
            all_fake.append(fake.cpu().numpy())
        
        all_real = np.concatenate(all_real)
        all_fake = np.concatenate(all_fake)
        
        cm = ConfusionMatrix(actual_vector=all_real,predict_vector=all_fake)
        acc = cm.Overall_ACC
        cen = cm.Overall_CEN
        mcc = cm.Overall_MCC
        return acc,cen,mcc