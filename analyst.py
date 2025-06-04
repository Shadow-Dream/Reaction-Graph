import utils
import argparse

parser = argparse.ArgumentParser("Analyst")
parser.add_argument("--graph_type", type=str,default="reaction_graph")
parser.add_argument("--reactions", type=str, nargs='+',default=["C1CNCCN1.ClCCBr>>CCN1CCNCC1Cl"])
args = parser.parse_args()

analyst_class = utils.get_analyst()
analyst = analyst_class()
metadata = analyst(args.reactions)
utils.print_metadata(metadata)