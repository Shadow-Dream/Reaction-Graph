import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = int, default = 0)
parser.add_argument("--reactant",type=str,default="")
parser.add_argument("--reagent",type=str,default="")
parser.add_argument("--product",type=str,default="")
parser.add_argument("--workdir",type=str,default="")
args = parser.parse_args()

subprocess.run([
    "python", f"inference_uspto_gram.py",
    "--reactant", args.reactant,
    "--reagent", args.reagent,
    "--product", args.product,
    "--workdir", args.workdir
])

subprocess.run([
    "python", f"inference_uspto_subgram.py",
    "--reactant", args.reactant,
    "--reagent", args.reagent,
    "--product", args.product,
    "--workdir", args.workdir
])

subprocess.run([
    "python", f"inference_gram.py",
    "--reactant", args.reactant,
    "--reagent", args.reagent,
    "--product", args.product,
    "--workdir", args.workdir
])

subprocess.run([
    "python", f"inference_subgram.py",
    "--reactant", args.reactant,
    "--reagent", args.reagent,
    "--product", args.product,
    "--workdir", args.workdir
])
