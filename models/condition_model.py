from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from networks import ConditionNetwork
import os
import time
from metadatas import DEFAULT_CONDITION_CONFIG,ConfigLoader
from datetime import datetime

class ConditionModel:
    def __init__(self,
                 dataloader_train = None,
                 dataloader_test = None,
                 dataloader_val = None,
                 config = {},
                 **kwargs
                 ):
        config_loader = ConfigLoader(DEFAULT_CONDITION_CONFIG,config,kwargs)
        config_loader.apply(self)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_val = dataloader_val

        self.init_task()
        self.init_network()
        self.init_criterion()

    def init_task(self):
        self.model_dir = os.path.join(self.model_dir,self.experiment_id)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir,exist_ok=True)
        self.split_lengths = [0,self.dim_catalyst,self.dim_solvent,self.dim_solvent,self.dim_reagent,self.dim_reagent]
        self.split_lengths = [sum(self.split_lengths[:i+1]) for i in range(6)]

    def init_network(self):
        self.model = ConditionNetwork(
            self.dim_catalyst,
            self.dim_solvent,
            self.dim_reagent,
            **self.network_config).to(self.device)
        
    def init_criterion(self):
        self.criterion = CrossEntropyLoss(reduce="none")

        self.pretrain_optimizer = Adam(self.model.parameters(),**self.pretrain_optimizer_config)
        self.pretrain_scheduler = ReduceLROnPlateau(self.pretrain_optimizer,**self.pretrain_scheduler_config)

        finetune_parameters = []
        finetune_parameters += list(self.model.inputs.parameters())
        finetune_parameters += list(self.model.hiddens.parameters())
        finetune_parameters += list(self.model.outputs.parameters())
        finetune_parameters += list(self.model.reaction_nn.sparsify.parameters())
        self.finetune_optimizer = Adam(finetune_parameters,**self.finetune_optimizer_config)
        self.finetune_scheduler = ReduceLROnPlateau(self.finetune_optimizer,**self.finetune_scheduler_config)
        
        self.learning_rates = {}
        none_weights = self.criterion_config["none_weights"]
        self.learning_rates["catalyst1"] = torch.ones([self.dim_catalyst]).to(self.device)
        self.learning_rates["catalyst1"][self.catalyst_none_index] = none_weights["catalyst1"]
        self.learning_rates["solvent1"] = torch.ones([self.dim_solvent]).to(self.device)
        self.learning_rates["solvent1"][self.solvent_none_index] = none_weights["solvent1"]
        self.learning_rates["solvent2"] = torch.ones([self.dim_solvent]).to(self.device)
        self.learning_rates["solvent2"][self.solvent_none_index] = none_weights["solvent2"]
        self.learning_rates["reagent1"] = torch.ones([self.dim_reagent]).to(self.device)
        self.learning_rates["reagent1"][self.reagent_none_index] = none_weights["reagent1"]
        self.learning_rates["reagent2"] = torch.ones([self.dim_reagent]).to(self.device)
        self.learning_rates["reagent2"][self.reagent_none_index] = none_weights["reagent2"]

    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.criterion_config["max_gradient"])

    def load(self,filename):
        state_dict = torch.load(filename)
        log = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model state dict from {filename}. Load log: {log}")

    def save(self,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".ckpt"
        torch.save(self.model.state_dict(),f"{self.model_dir}/{model_name}")

    def get_category_learning_rate(self,real,category):
        return (self.learning_rates[category]*real).sum(-1)
    
    def freeze_parameters(self):
        for param in self.model.reaction_nn.parameters():
            param.requires_grad = False
        for param in self.model.reaction_nn.sparsify.parameters():
            param.requires_grad = True

    def reset_parameters(self):
        for param in self.model.inputs.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.model.hiddens.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.model.outputs.parameters():
            nn.init.normal_(param,std=0.1)
        for param in self.model.reaction_nn.sparsify.parameters():
            nn.init.normal_(param,std=0.1)

    def train(self,**kwargs):
        pretrain_epoches = kwargs["pretrain_epoches"] if "pretrain_epoches" in kwargs else self.training_config["pretrain_epoches"]
        finetune_epoches = kwargs["finetune_epoches"] if "finetune_epoches" in kwargs else self.training_config["finetune_epoches"]
        progress_bar = kwargs["progress_bar"] if "progress_bar" in kwargs else self.progress_bar
        smoothing = kwargs["smoothing"] if "smoothing" in kwargs else self.criterion_config["smoothing"]
        accumulation_steps = kwargs["accumulation_steps"] if "accumulation_steps" in kwargs else self.criterion_config["accumulation_steps"]
        loss_display_buffer = kwargs["loss_display_buffer"] if "loss_display_buffer" in kwargs else self.loss_display_buffer
        log_delta = kwargs["log_delta"] if "log_delta" in kwargs else self.log_delta
        save_delta = kwargs["save_delta"] if "save_delta" in kwargs else self.training_config["save_delta"]

        best_accuracy = 0

        print("Begin Training.")

        for epoch in range(pretrain_epoches + finetune_epoches):
            if epoch == pretrain_epoches:
                self.freeze_parameters()
                self.reset_parameters()

            self.model.train()
            progress = tqdm(enumerate(self.dataloader_train),total=len(self.dataloader_train)) if progress_bar else enumerate(self.dataloader_train)
            losses = []
            for index,batch in progress:
                inputs = batch["input"]
                outputs = batch["output"]
                real_catalyst1,real_solvent1,real_solvent2,real_reagent1,real_reagent2 = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
                features = self.model.embedding(inputs)
                if epoch < pretrain_epoches:
                    learning_rate_catalyst1 = self.get_category_learning_rate(real_catalyst1,"catalyst1")
                    learning_rate_solvent1 = self.get_category_learning_rate(real_solvent1,"solvent1")
                    learning_rate_solvent2 = self.get_category_learning_rate(real_solvent2,"solvent2")
                    learning_rate_reagent1 = self.get_category_learning_rate(real_reagent1,"reagent1")
                    learning_rate_reagent2 = self.get_category_learning_rate(real_reagent2,"reagent2")
                    smooth_real_catalyst1 = real_catalyst1 * smoothing[0] + torch.ones_like(real_catalyst1) * (1 - smoothing[0]) / real_catalyst1.shape[-1]
                    smooth_real_solvent1 = real_solvent1 * smoothing[1] + torch.ones_like(real_solvent1) * (1 - smoothing[1]) / real_solvent1.shape[-1]
                    smooth_real_solvent2 = real_solvent2 * smoothing[2] + torch.ones_like(real_solvent2) * (1 - smoothing[2]) / real_solvent2.shape[-1]
                    smooth_real_reagent1 = real_reagent1 * smoothing[3] + torch.ones_like(real_reagent1) * (1 - smoothing[3]) / real_reagent1.shape[-1]
                    smooth_real_reagent2 = real_reagent2 * smoothing[4] + torch.ones_like(real_reagent2) * (1 - smoothing[4]) / real_reagent2.shape[-1]
                    loss_catalyst1 = self.criterion(self.model.catalyst1(features),smooth_real_catalyst1) * learning_rate_catalyst1
                    loss_solvent1 = self.criterion(self.model.solvent1(features,real_catalyst1),smooth_real_solvent1) * learning_rate_solvent1
                    loss_solvent2 = self.criterion(self.model.solvent2(features,real_catalyst1,real_solvent1),smooth_real_solvent2) * learning_rate_solvent2
                    loss_reagent1 = self.criterion(self.model.reagent1(features,real_catalyst1,real_solvent1,real_solvent2),smooth_real_reagent1) * learning_rate_reagent1
                    loss_reagent2 = self.criterion(self.model.reagent2(features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),smooth_real_reagent2) * learning_rate_reagent2
                else:
                    loss_catalyst1 = self.criterion(self.model.catalyst1(features),real_catalyst1)
                    loss_solvent1 = self.criterion(self.model.solvent1(features,real_catalyst1),real_solvent1)
                    loss_solvent2 = self.criterion(self.model.solvent2(features,real_catalyst1,real_solvent1),real_solvent2)
                    loss_reagent1 = self.criterion(self.model.reagent1(features,real_catalyst1,real_solvent1,real_solvent2),real_reagent1)
                    loss_reagent2 = self.criterion(self.model.reagent2(features,real_catalyst1,real_solvent1,real_solvent2,real_reagent1),real_reagent2)
                loss = loss_catalyst1.mean() + loss_solvent1.mean() + loss_solvent2.mean() + loss_reagent1.mean() + loss_reagent2.mean()
                loss /= accumulation_steps
                loss.backward()

                if (index + 1) % accumulation_steps == 0:
                    self.postprocess_gradient(self.model.parameters())
                    if epoch < pretrain_epoches:
                        self.pretrain_optimizer.step()
                        self.pretrain_optimizer.zero_grad()
                    else:
                        self.finetune_optimizer.step()
                        self.finetune_optimizer.zero_grad()
                
                loss = float(loss.detach().cpu())
                losses.append(loss)
                losses = losses[-loss_display_buffer:]
                average_loss = float(np.array(losses).mean())
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    if index % log_delta == 0:
                        print(f"epoch:{epoch},iter:{index}/{len(self.dataloader_train)},loss:{average_loss}")
                    
            accuracy = self.validate(validate_type="val",detail_level="category")

            if epoch < pretrain_epoches:
                self.pretrain_scheduler.step(1 - accuracy["overall"][1])
            else:
                self.finetune_scheduler.step(1 - accuracy["overall"][1])

            if epoch < pretrain_epoches:
                print(f"epoch: {epoch}, learning rate: {self.pretrain_optimizer.param_groups[0]['lr']}")
            else:
                print(f"epoch: {epoch}, learning rate: {self.finetune_optimizer.param_groups[0]['lr']}")
            
            self.print_accuracy(accuracy)
            
            if epoch < pretrain_epoches:
                self.save_log(accuracy,self.pretrain_optimizer.param_groups[0]['lr'],f"-{epoch}")
            else:
                self.save_log(accuracy,self.finetune_optimizer.param_groups[0]['lr'],f"-{epoch}")

            current_accuracy = sum(accuracy["overall"].values())
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(f"-best")

            if epoch % save_delta == save_delta - 1:
                self.save(f"-{epoch}")
                self.save(f"-last")

        print("Training Done!")

    def print_accuracy(self,accuracy):
        topk_row="topk        "
        for topk in accuracy["overall"]:
            topk_row += " "  +(str(topk) + "      ")[:6]
        print(topk_row)
        for key in accuracy:
            accuracy_row = (key + "            ")[:12]
            for topk in accuracy[key]:
                accuracy_row += " " + (str(float(accuracy[key][topk])) + "      ")[:6]
            print(accuracy_row)

    def save_log(self,accuracy,learning_rate,postfix = ""):
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
    def inference(self, batch, metadata, **kwargs):
        beams = kwargs["beams"] if "beams" in kwargs else self.validate_config["beams"]
        self.model.eval()
        predictions = self.model.inference(batch,beams)
        predictions = predictions.cpu().numpy().tolist()
        results = []
        for prediction in predictions:
            conditions = []
            for condition in prediction:
                catalyst1, solvent1, solvent2, reagent1, reagent2 = condition
                catalyst1 = metadata["catalysts"][catalyst1]
                solvent1 = metadata["solvents"][solvent1]
                solvent2 = metadata["solvents"][solvent2]
                reagent1 = metadata["reagents"][reagent1]
                reagent2 = metadata["reagents"][reagent2]
                condition = [catalyst1, solvent1, solvent2, reagent1, reagent2]
                conditions.append(condition)
            results.append(conditions)
        return results

    @torch.no_grad()
    def validate(self,**kwargs):
        progress_bar = kwargs["progress_bar"] if "progress_bar" in kwargs else self.progress_bar
        validate_type = kwargs["validate_type"] if "validate_type" in kwargs else self.validate_config["validate_type"]
        metric = kwargs["metric"] if "metric" in kwargs else self.validate_config["metric"]
        beams = kwargs["beams"] if "beams" in kwargs else self.validate_config["beams"]
        detail_level = kwargs["detail_level"] if "detail_level" in kwargs else self.validate_config["detail_level"]

        print("Begin Validating.")

        self.model.eval()
        dataloader = self.dataloader_val if validate_type=="val" else self.dataloader_test if validate_type=="test" else self.dataloader_train
        
        keys = ["catalyst1","solvent1","solvent2","reagent1","reagent2"]
        accuracies = {key:{topk:[] for topk in metric} for key in keys + ["overall"]}
        progress = tqdm(dataloader,total=len(dataloader)) if progress_bar else dataloader

        if detail_level == "all":
            keys = []
            real_results = []
            fake_results = []


        for batch in progress:
            inputs = batch["input"]
            outputs = batch["output"]
            fake = self.model.inference(inputs,beams)
            outputs = [outputs[:,self.split_lengths[i]:self.split_lengths[i+1]] for i in range(5)]
            outputs = torch.cat([torch.argmax(outputs[i],-1).unsqueeze(-1) for i in range(5)],dim = -1)
            real = outputs
            error = (fake - real.unsqueeze(-2)).abs()
            if detail_level == "all":
                keys += batch["key"]
                real_results.append(real.cpu().numpy())
                fake_results.append(fake.cpu().numpy())

            for topk in metric:
                error_topk = error[:,:topk]
                error_overall_topk = error_topk.sum(-1).min(1).values
                errors_category_topk = error_topk.min(1).values
                accuracy_overall = (error_overall_topk <= 1e-7)
                accuarcy_category = (errors_category_topk <= 1e-7)
                accuracies["overall"][topk] += accuracy_overall.cpu().numpy().tolist()
                for index,key in enumerate(keys):
                    accuracies[key][topk] += accuarcy_category[:,index].cpu().numpy().tolist()

        print("Validation Done!")
        if detail_level == "all":
            real_results = np.concatenate(real_results)
            fake_results = np.concatenate(fake_results)
            results = {"key":keys,"real":real_results,"fake":fake_results,"result":accuracies}
            return results
        elif detail_level == "results":
            return accuracies
        elif detail_level == "category":
            for key in accuracies:
                for topk in metric:
                    accuracies[key][topk] = np.mean(accuracies[key][topk])
            return accuracies
        elif detail_level == "overall":
            return accuracies