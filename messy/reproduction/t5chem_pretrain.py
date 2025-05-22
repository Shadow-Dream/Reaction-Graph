from t5chem import run_trainer
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def set_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pistachio/pretrain/dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pistachio/pretrain/weights"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="pretrain"
    )
    parser.add_argument(
        "--pretrain",
        default='models'
    )
    parser.add_argument(
        "--vocab",
        default=''
    )
    parser.add_argument(
        "--tokenizer",
        default=''
    )
    parser.add_argument(
        "--random_seed",
        default=123,
        type=int
    )
    parser.add_argument(
        "--num_epoch",
        default=30,
        type=int
    )
    parser.add_argument(
        "--log_step",
        default=50000,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--init_lr",
        default=5e-4,
        type=float
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0
    )


parser = argparse.ArgumentParser()
set_args(parser)
args = parser.parse_args()
run_trainer.train(args)