import random
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gc


def load_data(file, source_col, target_col,):
    if isinstance(file, str):
        file_type = file.split(".")[-1]
        if file_type == "xlsx":
            df = pd.read_excel(file)
            
        elif file_type == "csv":
            df = pd.read_csv(file)
            
        elif file_type == "tsv":
            df = pd.read_csv(file, sep="\t")
            
        else:
            raise TypeError("file must be a excel or csv file")
            
    elif isinstance(file, pd.DataFrame):
        df = file
        
    else:
        raise TypeError("file must be a string or a pandas DataFrame")
    
    source = df[source_col].to_list()
    target = df[target_col].to_list()
    
    return source, target


def add_prefix(source, prefix = "corrupt: "):
    new_source = [prefix + x for x in source]
    
    return new_source


def data_pipeline(train_file, tokenizer, config, valid_file = None):
    train_source, train_target = load_data(train_file, config.source_col, config.target_col)
    valid_source, valid_target = load_data(valid_file, config.source_col, config.target_col)
    
    train_source = add_prefix(train_source, config.prefix)
    valid_source = add_prefix(valid_source, config.prefix)
    
    print("-"*10 + "Data Loaded!" + "-"*10)
    print("-"*10 + "Data Split complete!" + "-"*10)
    
    train_source = tokenizer(train_source, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    train_target = tokenizer(train_target, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    valid_source = tokenizer(valid_source, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    valid_target = tokenizer(valid_target, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    print("-"*10 + "Tokenizing Complete!" + "-"*10)
    
    train_target = train_target["input_ids"]
    valid_target = valid_target["input_ids"]
    
    train_target[train_target == tokenizer.pad_token_id] = -100
    valid_target[valid_target == tokenizer.pad_token_id] = -100
    
    train_dataset = T5Dataset(source = train_source, target = train_target)
    valid_dataset = T5Dataset(source = valid_source, target = valid_target)
    print("-"*10 + "Dataset initialized!" + "-"*10)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=8,
                                  pin_memory=True)
    
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=8,
                                  pin_memory=True)
    
    print("-"*10 + "DataLoader initialized!" + "-"*10)
    
    return train_dataloader, valid_dataloader
    

def load_test_dataloader(test_file, tokenizer, config):
    test_source, test_target = load_data(test_file, config.source_col, config.target_col)
    test_source = add_prefix(test_source, config.prefix)
    test_source = tokenizer(test_source, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    test_target = tokenizer(test_target, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    test_target = test_target["input_ids"]
    test_target[test_target == tokenizer.pad_token_id] = -100
    test_dataset = T5Dataset(source = test_source, target = test_target)
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=1,
                                  num_workers=8,
                                  pin_memory=True)
    return test_dataloader


class T5Dataset(Dataset):
    def __init__(self, source, target):
        self.source_id = source["input_ids"]
        self.source_mask = source["attention_mask"]
        self.target = target
        
    def __getitem__(self, idx):
        return self.source_id[idx], self.source_mask[idx], self.target[idx]
    
    def __len__(self):
        return len(self.source_id)
