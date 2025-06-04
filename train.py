import utils
import argparse
import json

parser = argparse.ArgumentParser("Train")
parser.add_argument("--dataset", type=str,default="uspto_condition")
parser.add_argument("--graph_type", type=str,default="reaction_graph")
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

if args.config:
    config = json.loads(args.config)
    utils.set_device(config)
    utils.set_seed(config)

    dataloader_class, model_class = utils.get_class(args.dataset,args.graph_type)
    dataloader_train = dataloader_class(dataset_type=f"Train",shuffle=True,**config)
    dataloader_val = dataloader_class(dataset_type=f"Val",shuffle=False,**config)
    model = model_class(dataloader_train, None, dataloader_val, config)
    
    model.train()
else:
    configs = utils.get_config(args.dataset,args.graph_type)
    for config in configs:
        utils.start_train(config, args)