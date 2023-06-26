import wandb
import torch

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertTokenizerFast, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from model import BertModel
from data import BertDataset, load_data, data_pipeline


def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        
        temp = config.device.split(",")
        devices = [int(x) for x in temp]
    
    
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_path)
    print("-"*10 + "Tokenizer initialized!" + "-"*10)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt")
    print("-"*10 + "Collator initialized!" + "-"*10)
    
    model  = BertModel(config=config)
    print("-"*10 + "Model initialized!" + "-"*10)
    
    train_dataloader, valid_dataloader = data_pipeline(config.data_path, tokenizer, collator, config)
    
    wandb_logger = WandbLogger(project=config.wandb_project, name=f"Bert-batch_size{config.batch_size}")
    wandb_logger.experiment.config["batch_size"] = config.batch_size
    print("-"*10 + "Wandb Setting Complete!" + "-"*10)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='./checkpoint',
                                          filename= f"Bert-batch_size{config.batch_size}"+'-{val_loss:.2f}',
                                          save_top_k=1,
                                          save_last=False,
                                          verbose=True,
                                          mode="min")
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        mode='min',
        patience=10,
    )
    
    trainer = pl.Trainer(devices=devices,
                         accelerator=accelerator,
                         precision=config.precision,
                        #  strategy=config.strategy,
                         enable_progress_bar=True,
                         callbacks=[checkpoint_callback, early_stopping],
                         max_steps=config.max_steps,
                         val_check_interval= 1000,
                         log_every_n_steps=1000,
                         num_sanity_val_steps=2,
                         logger=wandb_logger)
    print("-"*10 + "Train Start!" + "-"*10)
    
    trainer.fit(model, train_dataloader, valid_dataloader)
    print("-"*10 + "Train Finished!" + "-"*10)
    
    wandb.finish()
