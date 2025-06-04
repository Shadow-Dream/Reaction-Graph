import utils
import argparse

parser = argparse.ArgumentParser("Inference")
parser.add_argument("--dataset", type=str,default="uspto_condition")
parser.add_argument("--graph_type", type=str,default="reaction_graph")
parser.add_argument("--experiment", type=str,default="")
parser.add_argument("--checkpoint", type=str,default="")
parser.add_argument("--reactions", type=str, nargs='+',default=["C1CNCCN1.ClCCBr>>CCN1CCNCC1Cl"])
args = parser.parse_args()

config = utils.get_config(args.dataset,args.graph_type,args.experiment)
encoder_class, metadata = utils.get_encoder(args.dataset,args.graph_type,args.experiment)
dataloader_class, model_class = utils.get_class(args.dataset,args.graph_type)

utils.set_device(config)
utils.set_seed(config)
encoder = encoder_class(metadata)
dataloader = dataloader_class(load = False, **config)
model = model_class(config=config)

checkpoint = utils.get_checkpoint(config, args)
model.load(checkpoint)

batch = encoder(args.reactions)
batch = dataloader(batch)
results = model.inference(batch, metadata)
utils.print_results(args.reactions, results)