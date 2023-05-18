import pickle
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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
    
    return df


def split_source_target(df, source_col, target_col):
    source = []
    before_source = df[source_col].tolist()
    for i in range(len(before_source)):
        temp_source = [x + " " for x in before_source[i]]
        final_str = "".join(temp_source)
        final_str = final_str.rstrip(" ")
        source.append(final_str)
        
    target = df[target_col].tolist()
    label_index_dict = {}
    index_label_dict = {}
    
    for i, label in enumerate(set(target)):
        label_index_dict[label] = i
        index_label_dict[i] = label
        
    with open("../data/label_index_dict.pkl", "wb") as f:
        pickle.dump(label_index_dict, f)

    with open("../data/index_label_dict.pkl", "wb") as f:
        pickle.dump(index_label_dict, f)

    return source, target


def load_pipeline(file, source_col, target_col):
    df = load_data(file)
    source, target = split_source_target(df, source_col, target_col)
    
    return source, target


def train_val_test_split(source, target, train_ratio = 0.8, valid_ratio = 0.1, test_ratio = 0.1, shuffle = True, stratified = True):
    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("train_ratio + valid_ratio + test_ratio must be 1")
    
    if stratified:
        train_source, valid_source, train_target, valid_target = train_test_split(source, target, train_size = train_ratio, shuffle = shuffle, stratify = target)
        valid_source, test_source, valid_target, test_target = train_test_split(valid_source, valid_target, test_size = test_ratio/(test_ratio + valid_ratio), shuffle = shuffle, stratify = valid_target)
        
    else:
        train_source, valid_source, train_target, valid_target = train_test_split(source, target, train_size = train_ratio, shuffle = shuffle)
        valid_source, test_source, valid_target, test_target = train_test_split(valid_source, valid_target, test_size = test_ratio/(test_ratio + valid_ratio), shuffle = shuffle)
        
    return (train_source, train_target), (valid_source, valid_target), (test_source, test_target)


class TCRBertDataset(Dataset):
    def __init__(self, source, target, tokenizer, max_length, task = "multi"):
        temp_source = tokenizer(source, padding = True, truncation = True, return_tensors = "pt", max_length = max_length)
        self.input_ids = temp_source.input_ids
        self.token_type_ids = temp_source.token_type_ids
        self.attention_mask = temp_source.attention_mask
        
        if task == "multi":
            label_index_dict = pickle.load(open("../data/label_index_dict.pkl", "rb"))
            temp_target = [label_index_dict[x] for x in target]
            
        else:
            temp_target = [1 if x == task else 0 for x in target]
            
        self.label = torch.LongTensor(temp_target)
        
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.label[idx]
    
    
    def __len__(self):
        return len(self.label)
        