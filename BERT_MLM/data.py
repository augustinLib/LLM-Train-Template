import pickle
import copy
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import gc

def load_data(file):
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
    
    data = df["AASeq"].to_list()
    
    return data


def data_pipeline(file, tokenizer, collator, config):
    data = load_data(file)
    print("-"*10 + "Data Loaded!" + "-"*10)
    
    train_data, valid_data = train_test_split(data, test_size=0.05, shuffle=True, random_state=42)
    del data
    gc.collect()
    print("-"*10 + "Data Split complete!" + "-"*10)
    
    train_data = tokenizer(train_data, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    valid_data = tokenizer(valid_data, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    print("-"*10 + "Tokenizing Complete!" + "-"*10)
    
    train_dataset = BertDataset(data = train_data)
    valid_dataset = BertDataset(data = valid_data)
    print("-"*10 + "Dataset initialized!" + "-"*10)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=16,
                                  pin_memory=True,
                                  collate_fn=collator,
                                  shuffle=True)
    
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=16,
                                  collate_fn=collator,
                                  pin_memory=True)
    
    print("-"*10 + "DataLoader initialized!" + "-"*10)
    
    return train_dataloader, valid_dataloader
    

class BertDataset(Dataset):
    def __init__(self, data):
       self.input_ids = torch.LongTensor(data.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx]
    
    def __len__(self):
        return len(self.input_ids)