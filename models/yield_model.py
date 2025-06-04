from torch.nn.modules.loss import CrossEntropyLoss,BCEWithLogitsLoss,MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import numpy as np
import torch
from tqdm import tqdm
from networks import YieldNetwork
import os
from metadatas import DEFAULT_YIELD_CONFIG,ConfigLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime

class YieldModel:
    def __init__(self,
                 dataloader_train = None,
                 dataloader_test = None,
                 dataloader_val = None,
                 config = {},
                 **kwargs
                 ):
        config_loader = ConfigLoader(DEFAULT_YIELD_CONFIG,config,kwargs)
        config_loader.apply(self)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test if dataloader_val is None else dataloader_val

        self.init_task()
        self.init_network()
        self.init_criterion()

    def init_task(self):
        self.model_dir = os.path.join(self.model_dir,self.experiment_id)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir,exist_ok=True)

    def init_network(self):
        self.model = YieldNetwork(**self.network_config).to(self.device)
        
    def init_criterion(self):
        self.criterion = MSELoss(reduction = 'none')
        self.optimizer = Adam(self.model.parameters(),**self.optimizer_config)
        self.scheduler = MultiStepLR(self.optimizer, **self.scheduler_config)

    def postprocess_gradient(self,parameters):
        for param in parameters:
            if param.grad is not None:
                param.grad[param.grad != param.grad] = 0
        torch.nn.utils.clip_grad_norm_(parameters, self.criterion_config["max_gradient"])

    def load(self,filename):
        state_dict = torch.load(filename)
        log = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model state dict from {filename}. Load log: {log}",flush = True)

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
        kl = kwargs["kl"] if "kl" in kwargs else self.criterion_config["kl"]
        best_accuracy = {"r2":0}

        print("Begin Training.",flush = True)
        
        for epoch in range(epoches):
            self.model.train()
            progress = tqdm(enumerate(self.dataloader_train),total=len(self.dataloader_train)) if progress_bar else enumerate(self.dataloader_train)
            losses = []
            for index,batch in progress:
                inputs = batch["input"]
                real_mean = batch["output"]
                real_mean = (real_mean - self.dataloader_train.mean) / self.dataloader_train.std
                pred_mean,pred_var = self.model(inputs)
                loss = self.criterion(pred_mean, real_mean)
                loss_mse = (1 - kl) * loss.mean()
                loss_kl = kl * (loss * torch.exp(-pred_var) + pred_var).mean()
                loss = loss_mse + loss_kl
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
                        print(f"epoch:{epoch},iter:{index}/{len(self.dataloader_train)},loss:{average_loss}",flush = True)
            self.scheduler.step()
            print(f"epoch: {epoch}, learning rate: {self.optimizer.param_groups[0]['lr']}",flush = True)
            
            accuracy = self.validate(validate_type = "test",num_inference_pass = 5,detail_level="results")

            if accuracy["r2"] > best_accuracy["r2"]:
                best_accuracy = accuracy
                self.save(f"-best")

            self.print_accuracy(accuracy)
            self.print_accuracy(best_accuracy)
            self.save_log(accuracy,self.optimizer.param_groups[0]['lr'],f"-{epoch}")
            self.save_log(best_accuracy,self.optimizer.param_groups[0]['lr'],f"-best")

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
            print(f"{key}: {value}",flush = True)

    @torch.no_grad()
    def inference(self, batch, metadata, **kwargs):
        num_inference_pass = kwargs["num_inference_pass"] if "num_inference_pass" in kwargs else self.validate_config["num_inference_pass"]
        
        self.model.eval()
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        mean_list = []
        var_list = []
        for _ in range(num_inference_pass):
            pred_mean, pred_logvar = self.model(batch)
            mean_list.append(pred_mean.cpu().numpy())
            var_list.append(np.exp(pred_logvar.cpu().numpy()))
        mean_list = np.concatenate(mean_list,axis=-1) * self.std + self.mean
        var_list = np.concatenate(var_list,axis=-1) * self.std ** 2
        predictions = np.mean(mean_list, -1).tolist()
        epistemic = np.var(mean_list, -1).tolist()
        aleatoric = np.mean(var_list, -1).tolist()
        results = list(zip(predictions,epistemic,aleatoric))
        return results

    @torch.no_grad()
    def validate(self,**kwargs):
        progress_bar = kwargs["progress_bar"] if "progress_bar" in kwargs else self.progress_bar
        detail_level = kwargs["detail_level"] if "detail_level" in kwargs else self.validate_config["detail_level"]
        num_inference_pass = kwargs["num_inference_pass"] if "num_inference_pass" in kwargs else self.validate_config["num_inference_pass"]
        print("Begin Validating.",flush = True)

        self.model.eval()
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        dataloader = self.dataloader_test
        progress = tqdm(dataloader,total=len(dataloader)) if progress_bar else dataloader

        keys = []
        real_results = []
        
        test_mean = []
        test_var = []

        for batch in progress:
            mean_list = []
            var_list = []
            inputs = batch["input"]
            for _ in range(num_inference_pass):
                pred_mean, pred_logvar = self.model(inputs)
                mean_list.append(pred_mean.cpu().numpy())
                var_list.append(np.exp(pred_logvar.cpu().numpy()))
            test_mean.append(np.concatenate(mean_list,axis=-1))
            test_var.append(np.concatenate(var_list,axis=-1))

            real = batch["output"].cpu().numpy().ravel()
            real_results.append(real)

            keys.append(batch["key"])
            
        test_mean = np.vstack(test_mean) * dataloader.std + dataloader.mean
        test_var = np.vstack(test_var) * dataloader.std ** 2
        real_results = np.concatenate(real_results)

        print("Validation Done!",flush = True)

        prediction = np.mean(test_mean, 1)
        epistemic = np.var(test_mean, 1)
        aleatoric = np.mean(test_var, 1)

        pred_mean = prediction
        real_mean = real_results
        mae = mean_absolute_error(real_mean, pred_mean)
        rmse = mean_squared_error(real_mean, pred_mean) ** 0.5
        r2 = r2_score(real_mean, pred_mean)

        if detail_level == "all":
            results = {
                "key":keys,
                "real":real_mean,
                "fake":pred_mean,
                "epistemic":epistemic,
                "aleatoric":aleatoric,
                "result":{
                    "mae":mae,
                    "rmse":rmse,
                    "r2":r2
                },
                }
            return results
        elif detail_level == "results":
            return {
                "mae":mae,
                "rmse":rmse,
                "r2":r2
            }
        elif detail_level == "r2":
            return r2