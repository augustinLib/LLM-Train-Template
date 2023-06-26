from argparse import ArgumentParser
from trainer import train
import os

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--wandb_project", type=str)
    # parser.add_argument("--wandb_run", type=str)
    # parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--device", default= -1, type=str)
    parser.add_argument("--precision", default= "bf16")
    # parser.add_argument("--precision", default= "mixed")
    # parser.add_argument("--strategy", default="deepspeed_stage_2")
    # parser.add_argument("--strategy", default="auto")
    parser.add_argument("--max_source_length", default= 512, type=int)
    parser.add_argument("--batch_size", default= 256, type=int)
    # parser.add_argument("--valid_batch_size", default= 512, type=int)
    parser.add_argument("--max_steps", default=10000000000000000000, type=int)
    parser.add_argument("--col_name", default= "data", type=str)
    
    args = parser.parse_args()
    
    return args


def main(config):
    train(config)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = parse_argument()
    main(config)