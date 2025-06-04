import dgl
import torch
import random
import numpy as np
import os
import dgl
import copy
import json
import subprocess
from deepmerge import always_merger
from datetime import datetime

from metadatas import (
    REACTION_GRAPH_USPTO_CONDITION_CONFIG, 
    REACTION_GRAPH_USPTO_TPL_CONFIG,
    REACTION_GRAPH_HTE_CONFIG,
    REACTION_GRAPH_USPTO_YIELD_CONFIG,
    USPTO_CONDITION_METADATA,
    USPTO_TPL_METADATA,
    USPTO_YIELD_GRAM_METADATA,
    USPTO_YIELD_SUBGRAM_METADATA,
    BUCHWALD_HARTWIG_METADATA,
    SUZUKI_MIYAURA_METADATA
)
from dataloaders.reaction_graph import (
    USPTOConditionDataloader,
    USPTOTPLDataloader,
    HTEDataloader,
    USPTOYieldDataloader
)
from models import (
    ConditionModel,
    TypeModel,
    YieldModel
)
from encoders import ReactionGraphEncoder
from analysts import ReactionGraphAnalyst

def batch(graphs):
    gdatas = [graph.gdata for graph in graphs]
    gdatas = {key:torch.cat([gdata[key] for gdata in gdatas]) for key in gdatas[0]}
    graphs = dgl.batch(graphs)
    graphs.gdata = gdatas
    return graphs

def set_seed(config):
    if "seed" in config:
        # The clumsy nn.LSTM ignores your seeds without setting this. Seriously slows down training!
        if "deterministic" in config and config["deterministic"]:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_device(config):
    if "gpu" in config:
        gpu = str(config["gpu"])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def get_analyst(graph_type):
    if graph_type == "reaction_graph":
        return ReactionGraphAnalyst

def get_class(dataset,graph_type):
    if graph_type == "reaction_graph":
        if dataset == "uspto_condition":
            dataloader_class = USPTOConditionDataloader
            model_class = ConditionModel
        elif dataset == "uspto_tpl":
            dataloader_class = USPTOTPLDataloader
            model_class = TypeModel
        elif dataset == "hte":
            dataloader_class = HTEDataloader
            model_class = YieldModel
        elif dataset == "uspto_yield":
            dataloader_class = USPTOYieldDataloader
            model_class = YieldModel
    return dataloader_class, model_class

def get_config(dataset,graph_type,selected_experiment_id = None):
    if graph_type == "reaction_graph":
        if dataset == "uspto_condition":
            config = REACTION_GRAPH_USPTO_CONDITION_CONFIG
        elif dataset == "uspto_tpl":
            config = REACTION_GRAPH_USPTO_TPL_CONFIG
        elif dataset == "hte":
            config = REACTION_GRAPH_HTE_CONFIG
        elif dataset == "uspto_yield":
            config = REACTION_GRAPH_USPTO_YIELD_CONFIG

    if "experiments" in config:
        configs = []
        experiments = config["experiments"]
        base_config = config
        for experiment_id, experiment in experiments.items():
            config = copy.deepcopy(base_config)
            always_merger.merge(config, experiment)
            config["experiment_id"] = experiment_id
            configs.append(config)
    else:
        config["experiment_id"] = "unnamed"
        configs = [config]

    if selected_experiment_id is not None:
        if selected_experiment_id == "":
            return configs[0]
        selected_configs = [
            config for config in configs 
            if config["experiment_id"] == selected_experiment_id
        ]
        config = selected_configs[0]
        return config
    
    return configs

def start_train(config, args):
    config["experiment_id"] += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    experiment_id = config["experiment_id"]
    config = json.dumps(config)
    command = ["nohup", "env", "PYTHONPATH=./", 
                "python", "train.py", 
                "--dataset",args.dataset,
                "--graph_type",args.graph_type,
                "--config",config]
    os.makedirs("logs/train",exist_ok=True)
    with open(f"logs/train/{args.graph_type}_{args.dataset}_{experiment_id}.log", "w") as log_file:
        subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Train {experiment_id} started")

def start_test(config, args):
    config["experiment_id"] += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    experiment_id = config["experiment_id"]
    config = json.dumps(config)
    checkpoint = config["checkpoint"]
    command = ["nohup", "env", "PYTHONPATH=./", 
                "python", "test.py", 
                "--dataset",args.dataset,
                "--graph_type",args.graph_type,
                "--checkpoint",checkpoint,
                "--config",config]
    os.makedirs("logs/test",exist_ok=True)
    with open(f"logs/test/{args.graph_type}_{args.dataset}_{experiment_id}.log", "w") as log_file:
        subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Test {experiment_id} started")

def print_results(keys, results):
    result_dict = {key:result for key,result in zip(keys, results)}
    result_string = json.dumps(result_dict, indent=4)
    print(result_string, flush=True)

def get_checkpoint(config, args):
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = config["checkpoint"]
    return checkpoint

def get_encoder(dataset,graph_type,selected_experiment_id = ""):
    if graph_type == "reaction_graph":
        encoder_class = ReactionGraphEncoder
    
    if dataset == "uspto_condition":
        encoder_config = USPTO_CONDITION_METADATA
    elif dataset == "uspto_tpl":
        encoder_config = USPTO_TPL_METADATA
    elif dataset == "uspto_yield":
        if "subgram" in selected_experiment_id:
            encoder_config = USPTO_YIELD_SUBGRAM_METADATA
        else:
            encoder_config = USPTO_YIELD_GRAM_METADATA
    elif dataset == "hte":
        if "suzuki_miyaura" in selected_experiment_id:
            encoder_config = SUZUKI_MIYAURA_METADATA
        else:
            encoder_config = BUCHWALD_HARTWIG_METADATA
    
    return encoder_class, encoder_config

def print_metadata(metadata):
    metadata_string = json.dumps(metadata,indent=4)
    print(metadata_string,flush=True)

def start_preprocess(args):
    dataset = args.dataset
    process_num = args.process_num
    progress_bar = args.progress_bar
    log_delta = args.log_delta
    batch_size = args.batch_size
    devices = [device for device in args.devices.split(",")]
    os.makedirs("logs/preprocess",exist_ok=True)
    if dataset=="uspto_condition":
        source_dir = args.source_dir
        target_dir = args.target_dir
        source_file = os.path.join(source_dir,"USPTO_condition.csv")
        target_name = "ReactionGraphUSPTOCondition"
        test_process_num = max(process_num * 1 // 10, 1)
        val_process_num = max(process_num * 1 // 10, 1)
        train_process_num = max(process_num - test_process_num - val_process_num, 1)
        device_id = 0
        print("Preprocess Begin.")
        for i in range(train_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_condition_preprocessor.py", 
                    "--source_file",source_file,
                    "--source_type","train",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(train_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_condition_preprocessor_train_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(test_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_condition_preprocessor.py", 
                    "--source_file",source_file,
                    "--source_type","test",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(test_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_condition_preprocessor_test_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(val_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_condition_preprocessor.py", 
                    "--source_file",source_file,
                    "--source_type","val",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(val_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_condition_preprocessor_val_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        print("The preprocessing subprocesses have been successfully started. The main process will exit.")
        print("The preprocessing progress will be displayed in the log files in ./logs/preprocess/.")
        print("You can monitor the status of the subprocess using tail, nvidia-smi or htop.")
    elif dataset=="uspto_tpl":
        source_dir = args.source_dir
        target_dir = args.target_dir
        source_file_train_val = os.path.join(source_dir,"uspto_1k_TPL_train_valid.tsv.gzip")
        source_file_test = os.path.join(source_dir,"uspto_1k_TPL_test.tsv.gzip")
        target_name = "ReactionGraphUSPTOTPL"
        test_process_num = max(process_num * 1 // 10, 1)
        val_process_num = max(process_num * 1 // 10, 1)
        train_process_num = max(process_num - test_process_num - val_process_num, 1)
        device_id = 0
        print("Preprocess Begin.")
        for i in range(train_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_tpl_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","train",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(train_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_tpl_preprocessor_train_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(test_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_tpl_preprocessor.py", 
                    "--source_file",source_file_test,
                    "--source_type","test",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(test_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_tpl_preprocessor_test_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(val_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_tpl_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","val",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(val_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_tpl_preprocessor_val_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        print("The preprocessing subprocesses have been successfully started. The main process will exit.")
        print("The preprocessing progress will be displayed in the log files in ./logs/preprocess/.")
        print("You can monitor the status of the subprocess using tail, nvidia-smi or htop.")
    elif dataset == "hte":
        datasets = ["buchwald_hartwig"] * 14 + ["suzuki_miyaura"] * 10
        splits = [f"split{i+1}" for i in range(10)]
        splits += [f"test{i+1}" for i in range(4)]
        splits += [f"split{i+1}" for i in range(10)]
        device_id = 0
        print("Preprocess Begin.")
        for dataset, split in zip(datasets, splits):
            source_dir = os.path.join(args.source_dir,dataset,split)
            target_dir = os.path.join(args.target_dir,dataset,split)
            source_file_train = os.path.join(source_dir, "train.csv")
            source_file_test = os.path.join(source_dir, "test.csv")
            dataset = dataset.replace('_', ' ').title().replace(' ', '')
            target_name = f"ReactionGraph{dataset}{split.capitalize()}"
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/hte_preprocessor.py", 
                    "--source_file",source_file_train,
                    "--source_type","train",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_hte_{split}_preprocessor_train.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/hte_preprocessor.py", 
                    "--source_file",source_file_test,
                    "--source_type","test",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_hte_{split}_preprocessor_test.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)
        print("The preprocessing subprocesses have been successfully started. The main process will exit.")
        print("The preprocessing progress will be displayed in the log files in ./logs/preprocess/.")
        print("You can monitor the status of the subprocess using tail, nvidia-smi or htop.")
    elif dataset == "uspto_yield":
        source_dir = args.source_dir
        base_target_dir = args.target_dir
        total_process_num = process_num
        device_id = 0

        process_num = int(total_process_num * 0.4)
        source_file_train_val = os.path.join(source_dir,"gram","gram_train_random_split.tsv")
        source_file_test = os.path.join(source_dir,"gram","gram_test_random_split.tsv")
        target_dir = os.path.join(base_target_dir,"gram")
        target_name = "ReactionGraphUSPTOYieldGram"
        test_process_num = max(process_num * 2 // 10, 1)
        val_process_num = max(process_num * 1 // 10, 1)
        train_process_num = max(process_num - test_process_num - val_process_num, 1)

        for i in range(train_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","train",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(train_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_gram_preprocessor_train_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(val_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","val",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(val_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_gram_preprocessor_val_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)
        
        for i in range(test_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_test,
                    "--source_type","test",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(test_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_gram_preprocessor_test_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        process_num = total_process_num - process_num
        source_file_train_val = os.path.join(source_dir,"subgram","milligram_train_random_split.tsv")
        source_file_test = os.path.join(source_dir,"subgram","milligram_test_random_split.tsv")
        target_dir = os.path.join(base_target_dir,"subgram")
        target_name = "ReactionGraphUSPTOYieldSubgram"
        test_process_num = max(process_num * 2 // 10, 1)
        val_process_num = max(process_num * 1 // 10, 1)
        train_process_num = max(process_num - test_process_num - val_process_num, 1)

        for i in range(train_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","train",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(train_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_subgram_preprocessor_train_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        for i in range(val_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_train_val,
                    "--source_type","val",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(val_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_subgram_preprocessor_val_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)
        
        for i in range(test_process_num):
            command = ["nohup", "env", "PYTHONPATH=./", 
                    "python", "preprocessors/reaction_graph/uspto_yield_preprocessor.py", 
                    "--source_file",source_file_test,
                    "--source_type","test",
                    "--target_dir",target_dir,
                    "--target_name",target_name,
                    "--progress_bar","true" if progress_bar else "false",
                    "--log_delta",str(log_delta),
                    "--split_num",str(test_process_num),
                    "--split_index",str(i),
                    "--batch_size",str(batch_size),
                    "--device",devices[device_id]]
            with open(f"logs/preprocess/reaction_graph_uspto_yield_subgram_preprocessor_test_{i}.log", "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            device_id = (device_id + 1) % len(devices)

        print("The preprocessing subprocesses have been successfully started. The main process will exit.")
        print("The preprocessing progress will be displayed in the log files in ./logs/preprocess/.")
        print("You can monitor the status of the subprocess using tail, nvidia-smi or htop.")