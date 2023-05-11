from argparse import ArgumentParser
from trainer import train
import os

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="None", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", default="None", type=str)
    parser.add_argument("--wandb_project", type=str)
    # parser.add_argument("--wandb_run", type=str)
    # parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--device", default= -1, type=str)
    # parser.add_argument("--device", default= 2, type=int)
    # parser.add_argument("--precision", default= "bf16")
    # parser.add_argument("--precision", default= "mixed")
    parser.add_argument("--precision",default=16, type=int)
    parser.add_argument("--strategy", default="deepspeed_stage_2")
    # parser.add_argument("--strategy", default="ddp")
    # parser.add_argument("--strategy", default="auto")
    parser.add_argument("--max_source_length", default= 512, type=int)
    parser.add_argument("--batch_size", default= 256, type=int)
    parser.add_argument("--max_steps", default=10000000000000000000, type=int)
    parser.add_argument("--val_check_interval", default= 1000, type=int)
    parser.add_argument("--log_every_n_steps", default= 1000, type=int)
    parser.add_argument("--num_sanity_val_steps", default= 2, type=int)
    parser.add_argument("--num_warmup_steps", default= 10000, type=int)
    parser.add_argument("--vocab_size", type=int)
    
    args = parser.parse_args()
    
    return args


def main(config):
    train(config)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = parse_argument()
    main(config)