import random
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
    
    data = df["record_value"].to_list()
    
    return data


def data_pipeline(train_file, tokenizer, collator, config, valid_file = None):
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    
    print("-"*10 + "Data Loaded!" + "-"*10)
    print("-"*10 + "Data Split complete!" + "-"*10)
    
    train_data = tokenizer(train_data, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    valid_data = tokenizer(valid_data, padding = True, truncation = True, return_tensors = "pt", max_length = config.max_source_length)
    print("-"*10 + "Tokenizing Complete!" + "-"*10)
    
    train_dataset = T5Dataset(data = train_data)
    valid_dataset = T5Dataset(data = valid_data)
    print("-"*10 + "Dataset initialized!" + "-"*10)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=8,
                                  pin_memory=True,
                                  collate_fn=collator,
                                  shuffle=True)
    
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=config.batch_size,
                                  num_workers=8,
                                  collate_fn=collator,
                                  pin_memory=True)
    
    print("-"*10 + "DataLoader initialized!" + "-"*10)
    
    return train_dataloader, valid_dataloader
    

class T5Dataset(Dataset):
    def __init__(self, data):
       self.input_ids = torch.LongTensor(data.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx]
    
    def __len__(self):
        return len(self.input_ids)


import random
import torch
from torch.nn.utils.rnn import pad_sequence

class T5SpanCorruptCollator():
    def __init__(self, tokenizer, span_length=3):
        self.tokenizer = tokenizer
        self.span_length = span_length
        
        
    def __call__(self, batch_corpus):
        return self.span_corrupt(batch_corpus)
    
    
    def span_corrupt(self, corpus):
        input_zip = ()
        label_zip = ()
        for i in corpus:
            input, label =self._span_corrupt(i)
            input_zip += (input,)
            label_zip += (label,)

        source_corpus = pad_sequence(input_zip, batch_first=True)
        target_corpus = pad_sequence(label_zip, batch_first=True)
        target_corpus[target_corpus == 0] = int(-100)
        
        batch = {
            "input_ids": source_corpus,
            "labels": target_corpus
        }
        
        return batch
        
        
    def _span_corrupt(self, text):
        text = text[:torch.where(text == 1)[0]]
        position = 0
        masked_num = 0
        sentinel_map = sorted(self.tokenizer.get_sentinel_token_ids(), reverse=True)
        source = torch.tensor([], dtype=torch.long)
        target = torch.tensor([], dtype=torch.long)

        span_scale = self.span_length // 2 if self.span_length % 2 == 0 else (self.span_length-1) // 2

        while (position < len(text)):
            ismask = random.random() <= 0.15
            # masking 안되는 85%
            if not ismask:
                source = torch.cat([source, torch.tensor([text[position]], dtype=torch.long)])
                position += 1
                continue
            
            # masking되는 15%
            # source에 mask안된 token 추가
            source = torch.cat([source, torch.tensor([sentinel_map[masked_num]], dtype=torch.long)])
            target = torch.cat([target, torch.tensor([sentinel_map[masked_num]], dtype=torch.long)])

            position += 1

            if position-span_scale < 0:
                target = torch.cat([target, text[:position+span_scale+1]])

            elif position+span_scale+1 > len(text):
                target = torch.cat([target, text[position-span_scale:]])

            else:
                target = torch.cat([target, text[position-span_scale:position+span_scale+1]])

            position += (span_scale+1)
            masked_num += 1

        source = torch.cat([source, torch.tensor([1], dtype=torch.long)])
        target = torch.cat([target, torch.tensor([1], dtype=torch.long)])

        return source, target