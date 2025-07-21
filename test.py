import utils
import argparse
import json

parser = argparse.ArgumentParser("Test")
parser.add_argument("--dataset", type=str,default="uspto_tpl")
parser.add_argument("--graph_type", type=str,default="reaction_graph")
parser.add_argument("--checkpoint", type=str,default="")
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

if args.config:
    config = json.loads(args.config)
    utils.set_device(config)
    utils.set_seed(config)

    dataloader_class, model_class = utils.get_class(args.dataset,args.graph_type)
    dataloader_test = dataloader_class(dataset_type=f"Test",shuffle=False,**config)
    model = model_class(None, dataloader_test, None, config)
    model.load(args.checkpoint)
    accuracy = model.validate(validate_type = "test")
    model.print_accuracy(accuracy)

else:
    configs = utils.get_config(args.dataset,args.graph_type)
    for config in configs:
        utils.start_test(config, args)