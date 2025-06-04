import argparse
import utils

parser = argparse.ArgumentParser("Preprocessor")
parser.add_argument("--dataset", type=str,default="uspto_condition")
parser.add_argument("--source_dir", type=str,default="datasets/uspto_condition")
parser.add_argument("--target_dir", type=str,default="datasets/uspto_condition")
parser.add_argument("--batch_size", type=int,default=32)
parser.add_argument("--process_num", type=int,default=40)
parser.add_argument("--progress_bar", type=bool, default=True)
parser.add_argument("--log_delta", type=int, default=100)
parser.add_argument("--devices",type=str,default="0,1,2,3,4,5,6,7")
args = parser.parse_args()

utils.start_preprocess(args)