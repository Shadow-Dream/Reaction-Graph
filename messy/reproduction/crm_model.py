
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
from torch.nn import functional as func
from tqdm import tqdm
from interfaces.CRM_interface import CRMInterface
from networks.CRM_feature_extractor import CRMFeatureExtractor
import itertools
from copy import deepcopy
class CRMModel:
    def __init__(self,
                 dataset_train = None,
                 dataset_test = None,
                 dataset_val = None,
                 model_dir = "",
                 device = "cuda",
                 dim_catalyst: int = 54,
                 dim_solvent: int = 87,
                 dim_reagent: int = 235,
                 parameters_for_model = {},
                 parameters_for_optimizer = {"lr":0.0005},
                 parameters_for_scheduler = {}):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val

        self.device = device
        self.model_dir = model_dir

        self.interface = CRMInterface()
        self.model = CRMFeatureExtractor(**parameters_for_model).to(device)

        self.cross_entropy_loss = CrossEntropyLoss()

        self.optimizer = Adam(self.model.parameters(),**parameters_for_optimizer)
        self.scheduler = ReduceLROnPlateau(self.optimizer,**parameters_for_scheduler)

        if self.dataset_train:
            self.dataset_train.search_for_condition = False
            self.dataset_train.device = device
        
        if self.dataset_val:
            self.dataset_val.search_for_condition = True
            self.dataset_val.encode_search_result = True
            self.dataset_val.device = device

        if self.dataset_test:
            self.dataset_test.search_for_condition = True
            self.dataset_test.encode_search_result = True
            self.dataset_test.device = device

        self.split_lengths = [0,dim_catalyst,dim_solvent,dim_solvent,dim_reagent,dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]

    def load(self,filename):
        self.model.load_state_dict(torch.load(f"{self.model_dir}/{filename}"))

    def save(self,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        torch.save(self.model.state_dict(),f"{self.model_dir}/{model_name}")

    def train(self,epoches = 100, save_delta = 10,progress_bar = True):
        for epoch in range(epoches):
            self.model.train()
            progress = tqdm(self.dataset_train,total=len(self.dataset_train)) if progress_bar else self.dataset_train
            losses = []
            for batch in progress:
                inputs = batch["inputs"]
                outputs = batch["outputs"]
                real_catalyst1,real_solvent1,real_solvent2,real_reagent1,real_reagent2 = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
                self.optimizer.zero_grad()
                features = self.model.embedding(inputs)
                loss_catalyst1 = self.cross_entropy_loss(self.model.catalyst1(features),real_catalyst1)
                loss_solvent1 = self.cross_entropy_loss(self.model.solvent1(features,real_catalyst1),real_solvent1)
                loss_solvent2 = self.cross_entropy_loss(self.model.solvent2(features,real_catalyst1,real_solvent1),real_solvent2)
                loss_reagent1 = self.cross_entropy_loss(self.model.reagent1(features,real_catalyst1,real_solvent1,real_solvent2),real_reagent1)
                loss_reagent2 = self.cross_entropy_loss(self.model.reagent2(features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),real_reagent2)
                loss = loss_catalyst1 + loss_solvent1 + loss_solvent2 + loss_reagent1 + loss_reagent2
                loss.backward()
                self.optimizer.step()
                loss = float(loss.detach().cpu())
                losses.append(loss)
                losses = losses if len(losses) <= 100 else losses[-100:]
                average_loss = float(np.array(losses).mean())
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    print(f"epoch:{epoch},loss:{average_loss}")
            accuracy = self.validate(progress_bar)
            self.scheduler.step(1 - accuracy["overall"][1])
            print(f"epoch: {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}")
            topk_row="topk        "
            for topk in accuracy["overall"]:
                topk_row += " "  +(str(topk) + "      ")[:6]
            print(topk_row)
            for key in accuracy:
                accuracy_row = (key + "            ")[:12]
                for topk in accuracy[key]:
                    accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
                print(accuracy_row)
            self.save_log(accuracy,f"-{epoch}")
            if epoch % save_delta == save_delta - 1:
                self.save(f"-{epoch}")
                self.save(f"-last")
                
    def save_log(self,accuracy,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        learning_rate = self.optimizer.param_groups[0]["lr"]
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
    def validate(self,
                 progress_bar = True,
                 type = "val",
                 metric = [1,3,5,10,15],
                 beams = [1,3,1,5,1]):
        self.model.eval()
        dataset = self.dataset_val if type=="val" else self.dataset_test if type=="test" else None
        
        keys = ["catalyst1","solvent1","solvent2","reagent1","reagent2"]
        accuracies = {key:{topk:[] for topk in metric} for key in keys + ["overall"]}
        progress = tqdm(dataset,total=len(dataset)) if progress_bar else dataset
        for batch in progress:
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            fake = self.model(inputs,beams)
            outputs = [[output[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)] for output in outputs]
            outputs = [torch.cat([torch.argmax(output[i],-1) for i in range(5)],dim=-1).reshape(-1,5) for output in outputs]
            real = outputs
            # [torch.cat([
            #     output[:,[0,1,2,3,4]],
            #     output[:,[0,2,1,3,4]],
            #     output[:,[0,1,2,4,3]],
            #     output[:,[0,2,1,4,3]],
            # ]) for output in outputs]

            for real_item,fake_item in zip(real,fake):
                error = abs(fake_item.unsqueeze(1) - real_item)
                for topk in metric:
                    error_overall_topk = error[:topk].reshape(-1,5).sum(-1).min(0).values
                    errors_category_topk = error[:topk].reshape(-1,5).min(0).values
                    accuracy_overall = (error_overall_topk <= 1e-7)
                    accuracies_categroy = (errors_category_topk <= 1e-7)
                    accuracies["overall"][topk].append(accuracy_overall.cpu().numpy().item())
                    for index,key in enumerate(keys):
                        accuracies[key][topk].append(accuracies_categroy[index].cpu().numpy().item())

        for key in accuracies:
            for topk in metric:
                accuracies[key][topk] = np.mean(accuracies[key][topk])
        return accuracies


class CRMModelForPistachio:
    def __init__(self,
                 dataset_train = None,
                 dataset_test = None,
                 dataset_val = None,
                 model_dir = "",
                 device = "cuda",
                 dim_catalyst: int = 70,
                 dim_solvent: int = 135,
                 dim_reagent: int = 274,
                 parameters_for_model = {},
                 parameters_for_optimizer = {"lr":0.0005},
                 parameters_for_scheduler = {}):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val

        self.device = device
        self.model_dir = model_dir

        self.interface = CRMInterface()
        self.model = CRMFeatureExtractor(
            dim_catalyst=dim_catalyst,
            dim_solvent=dim_solvent,
            dim_reagent=dim_reagent,
            **parameters_for_model
        ).to(device)

        self.cross_entropy_loss = CrossEntropyLoss()

        self.optimizer = Adam(self.model.parameters(),**parameters_for_optimizer)
        self.scheduler = ReduceLROnPlateau(self.optimizer,**parameters_for_scheduler)

        self.split_lengths = [0,dim_catalyst,dim_solvent,dim_solvent,dim_reagent,dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]

    def load(self,filename):
        self.model.load_state_dict(torch.load(f"{self.model_dir}/{filename}"))

    def save(self,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        torch.save(self.model.state_dict(),f"{self.model_dir}/{model_name}")

    def train(self,epoches = 100, save_delta = 10,progress_bar = True):
        for epoch in range(epoches):
            self.model.train()
            progress = tqdm(self.dataset_train,total=len(self.dataset_train)) if progress_bar else self.dataset_train
            losses = []
            for batch in progress:
                inputs = batch["inputs"]
                outputs = batch["outputs"]
                real_catalyst1,real_solvent1,real_solvent2,real_reagent1,real_reagent2 = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
                self.optimizer.zero_grad()
                features = self.model.embedding(inputs)
                loss_catalyst1 = self.cross_entropy_loss(self.model.catalyst1(features),real_catalyst1)
                loss_solvent1 = self.cross_entropy_loss(self.model.solvent1(features,real_catalyst1),real_solvent1)
                loss_solvent2 = self.cross_entropy_loss(self.model.solvent2(features,real_catalyst1,real_solvent1),real_solvent2)
                loss_reagent1 = self.cross_entropy_loss(self.model.reagent1(features,real_catalyst1,real_solvent1,real_solvent2),real_reagent1)
                loss_reagent2 = self.cross_entropy_loss(self.model.reagent2(features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),real_reagent2)
                loss = loss_catalyst1 + loss_solvent1 + loss_solvent2 + loss_reagent1 + loss_reagent2
                loss.backward()
                self.optimizer.step()
                loss = float(loss.detach().cpu())
                losses.append(loss)
                losses = losses if len(losses) <= 100 else losses[-100:]
                average_loss = float(np.array(losses).mean())
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    print(f"epoch:{epoch},loss:{average_loss}")
            accuracy = self.validate(progress_bar)
            self.scheduler.step(1 - accuracy["overall"][1])
            print(f"epoch: {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}")
            topk_row="topk        "
            for topk in accuracy["overall"]:
                topk_row += " "  +(str(topk) + "      ")[:6]
            print(topk_row)
            for key in accuracy:
                accuracy_row = (key + "            ")[:12]
                for topk in accuracy[key]:
                    accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
                print(accuracy_row)
            self.save_log(accuracy,f"-{epoch}")
            if epoch % save_delta == save_delta - 1:
                self.save(f"-{epoch}")
                self.save(f"-last")
                
    def save_log(self,accuracy,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        learning_rate = self.optimizer.param_groups[0]["lr"]
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
    def validate(self,
                 progress_bar = True,
                 type = "val",
                 metric = [1,3,5,10,15],
                 beams = [1,3,1,5,1]):
        self.model.eval()
        dataset = self.dataset_val if type=="val" else self.dataset_test if type=="test" else None
        
        keys = ["catalyst1","solvent1","solvent2","reagent1","reagent2"]
        accuracies = {key:{topk:[] for topk in metric} for key in keys + ["overall"]}
        progress = tqdm(dataset,total=len(dataset)) if progress_bar else dataset
        for batch in progress:
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            fake = self.model(inputs,beams)
            outputs = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
            outputs = torch.cat([torch.argmax(outputs[i],-1).unsqueeze(-1) for i in range(5)],dim = -1)
            real = outputs
            error = (fake - real.unsqueeze(-2)).abs()
            for topk in metric:
                error_topk = error[:,:topk]
                error_overall_topk = error_topk.sum(-1).min(1).values
                errors_category_topk = error_topk.min(1).values
                accuracy_overall = (error_overall_topk <= 1e-7)
                accuarcy_category = (errors_category_topk <= 1e-7)
                accuracies["overall"][topk] += accuracy_overall.cpu().numpy().tolist()
                for index,key in enumerate(keys):
                    accuracies[key][topk] += accuarcy_category[:,index].cpu().numpy().tolist()

        for key in accuracies:
            for topk in metric:
                accuracies[key][topk] = np.mean(accuracies[key][topk])
        return accuracies