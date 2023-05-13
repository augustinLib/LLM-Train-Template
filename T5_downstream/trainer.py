import wandb
import torch

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import deepspeed
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from model import T5ModelForPreTraining
from data import T5Dataset, data_pipeline, load_test_dataloader


def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        
        temp = config.device.split(",")
        devices = [int(x) for x in temp]
    
    tokenizer = T5TokenizerFast.from_pretrained(config.tokenizer_path)
    print("-"*10 + "Tokenizer initialized!" + "-"*10)
    
    model  = T5ModelForPreTraining(config=config, tokenizer=tokenizer)
    print("-"*10 + "Model initialized!" + "-"*10)
    
    train_dataloader, valid_dataloader = data_pipeline(config.train_data_path, tokenizer, config, valid_file=config.valid_data_path)
    
    wandb_logger = WandbLogger(project=config.wandb_project, name=f"T5-batch_size{config.batch_size}")
    wandb_logger.experiment.config["batch_size"] = config.batch_size
    print("-"*10 + "Wandb Setting Complete!" + "-"*10)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='./checkpoint',
                                          filename= f"T5-batch_size{config.batch_size}"+'-{val_loss:.2f}',
                                          save_top_k=1,
                                          save_last=False,
                                          verbose=True,
                                          mode="min")
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        mode='min',
        patience=10,
    )
    
    trainer = pl.Trainer(
                         accelerator=accelerator,
                         devices=devices,
                         precision=config.precision,
                        #  strategy=config.strategy,
                         enable_progress_bar=True,
                         callbacks=[checkpoint_callback, early_stopping],
                         max_steps=config.max_steps,
                         val_check_interval= config.val_check_interval,
                         log_every_n_steps=config.log_every_n_steps,
                         num_sanity_val_steps=config.num_sanity_val_steps,
                         logger=wandb_logger
                         )
    
    print("-"*10 + "Train Start!" + "-"*10)
    
    trainer.fit(model, train_dataloader, valid_dataloader)
    print("-"*10 + "Train Finished!" + "-"*10)

    test_dataloader=  load_test_dataloader(config.test_data_path, tokenizer, config)
    trainer.test(model, test_dataloader)
    
    wandb.finish()