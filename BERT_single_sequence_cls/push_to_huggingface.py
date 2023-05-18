from argparse import ArgumentParser
from model import BertModel
from transformers import BertTokenizerFast

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", default= "./", type=str)
    parser.add_argument("--checkpoint_model_path", default= "./", type=str)
    parser.add_argument("--model_save_path", default= "./", type=str)
    parser.add_argument("--huggingface_path", default= "./", type=str)

    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    config = parse_argument()
    model = BertModel.load_from_checkpoint(config.checkpoint_model_path)
    model.model.save_pretrained(config.model_save_path)
    model.model.push_to_hub(config.huggingface_path)

    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_path)
    tokenizer.push_to_hub(config.huggingface_path)