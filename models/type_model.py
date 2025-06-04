from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from networks import TypeNetwork
import os
import datetime
from metadatas import DEFAULT_TYPE_CONFIG,ConfigLoader
import pycm

class TypeModel:
    def __init__(self,
                 dataloader_train = None,
                 dataloader_test = None,
                 dataloader_val = None,
                 config = {},
                 **kwargs
                 ):
        config_loader = ConfigLoader(DEFAULT_TYPE_CONFIG,config,kwargs)
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

    def init_network(self):
        self.model = TypeNetwork(
            self.num_types,
            **self.network_config).to(self.device)
        
    def init_criterion(self):
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(),**self.optimizer_config)
        self.scheduler = ReduceLROnPlateau(self.optimizer,**self.scheduler_config)

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

    def train(self,**kwargs):
        epoches = kwargs["epoches"] if "epoches" in kwargs else self.training_config["epoches"]
        progress_bar = kwargs["progress_bar"] if "progress_bar" in kwargs else self.progress_bar
        accumulation_steps = kwargs["accumulation_steps"] if "accumulation_steps" in kwargs else self.criterion_config["accumulation_steps"]
        loss_display_buffer = kwargs["loss_display_buffer"] if "loss_display_buffer" in kwargs else self.loss_display_buffer
        log_delta = kwargs["log_delta"] if "log_delta" in kwargs else self.log_delta
        save_delta = kwargs["save_delta"] if "save_delta" in kwargs else self.training_config["save_delta"]
        
        best_accuracy = 0

        print("Begin Training.")
        
        for epoch in range(epoches):
            self.model.train()
            progress = tqdm(enumerate(self.dataloader_train),total=len(self.dataloader_train)) if progress_bar else self.dataloader_train
            losses = []
            for index,batch in progress:
                inputs = batch["input"]
                real = batch["output"]
                fake = self.model(inputs)
                loss = self.criterion(fake,real)
                loss /= accumulation_steps
                loss.backward()
                if (index + 1) % accumulation_steps==0:
                    self.postprocess_gradient(self.model.parameters())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                loss = float(loss.detach().cpu())
                losses.append(loss)
                losses = losses[-loss_display_buffer:]
                average_loss = float(np.array(losses).mean())
                if progress_bar:
                    progress.set_postfix({"epoch":epoch,"loss":average_loss})
                else:
                    if index % log_delta == 0:
                        print(f"epoch:{epoch},iter:{index}/{len(self.dataloader_train)},loss:{average_loss}")
            accuracy = self.validate(validate_type="val",detail_level="results")
            print(f"epoch: {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}")
            self.print_accuracy(accuracy)
            self.save_log(accuracy,self.optimizer.param_groups[0]['lr'],f"-{epoch}")

            accuracy = accuracy["acc"]
            self.scheduler.step(1-accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save(f"-best")

            if epoch % save_delta == save_delta - 1:
                self.save(f"-{epoch}")
                self.save(f"-last")

    def save_log(self,accuracy,learning_rate,postfix = ""):
        model_name = self.__class__.__name__.lower()
        model_name += postfix
        model_name += ".log"
        with open(f"{self.model_dir}/{model_name}","w") as f:
            f.write(f"learning rate: {learning_rate}, acc: {accuracy}")

    def print_accuracy(self,accuracy):
        for key,value in accuracy.items():
            print(f"{key}: {value}")

    @torch.no_grad()
    def inference(self, batch, metadata, **kwargs):
        self.model.eval()
        predictions = self.model(batch).argmax(-1)
        predictions = predictions.cpu().numpy().tolist()
        return predictions

    @torch.no_grad()
    def validate(self,**kwargs):
        progress_bar = kwargs["progress_bar"] if "progress_bar" in kwargs else self.progress_bar
        validate_type = kwargs["validate_type"] if "validate_type" in kwargs else self.validate_config["validate_type"]
        detail_level = kwargs["detail_level"] if "detail_level" in kwargs else self.validate_config["detail_level"]

        print("Begin Validating.")

        self.model.eval()
        dataloader = self.dataloader_val if validate_type=="val" else self.dataloader_test if validate_type=="test" else self.dataloader_train
        progress = tqdm(dataloader,total=len(dataloader)) if progress_bar else dataloader

        keys = []
        real_results = []
        fake_results = []

        for batch in progress:
            inputs = batch["input"]
            outputs = batch["output"]
            real = outputs.argmax(-1)
            fake = self.model(inputs).argmax(-1)
            keys.append(batch["key"])
            real_results.append(real)
            fake_results.append(fake)
            
        real_results = torch.cat(real_results).cpu().numpy().tolist()
        fake_results = torch.cat(fake_results).cpu().numpy().tolist()

        cm = pycm.ConfusionMatrix(actual_vector=real_results, predict_vector=fake_results)
        print("Validation Done!")
        if detail_level == "all":
            results = {"key":keys,"real":real_results,"fake":fake_results,"result":{
                "acc":cm.Overall_ACC,
                "cen":cm.Overall_CEN,
                "mcc":cm.Overall_MCC,
                "f1":cm.F1_Macro
            }}
            return results
        elif detail_level == "results":
            return {
                "acc":cm.Overall_ACC,
                "cen":cm.Overall_CEN,
                "mcc":cm.Overall_MCC,
                "f1":cm.F1_Macro
            }
        elif detail_level == "accuracy":
            return cm.Overall_ACC