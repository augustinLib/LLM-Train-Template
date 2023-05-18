import torch
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, softmax


class BertModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(config.data_path, num_labels=config.num_labels)
        self.save_hyperparameters()
        
        if config.task == "multi":
            self.train_acc = Accuracy(task="multiclass", multiclass=True, num_classes=config.num_labels)
            self.valid_acc = Accuracy(task="multiclass", multiclass=True, num_classes=config.num_labels)
            self.test_acc = Accuracy(task="multiclass", multiclass=True, num_classes=config.num_labels)
            
            self.train_f1 = F1Score(task="multiclass", multiclass=True, num_classes=config.num_labels, average="macro")
            self.valid_f1 = F1Score(task="multiclass", multiclass=True, num_classes=config.num_labels, average="macro")
            self.test_f1 = F1Score(task="multiclass", multiclass=True, num_classes=config.num_labels, average="macro")
        
            
        else:
            self.train_acc = Accuracy(task="binary", num_classes=config.num_labels)
            self.valid_acc = Accuracy(task="binary", num_classes=config.num_labels)
            self.test_acc = Accuracy(task="binary", num_classes=config.num_labels)
            
            self.train_f1 = F1Score(task="binary", num_classes=config.num_labels)
            self.valid_f1 = F1Score(task="binary", num_classes=config.num_labels)
            self.test_f1 = F1Score(task="binary", num_classes=config.num_labels)

        
        self.config = config
        
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        # (batch_size, sequence_length, config.vocab_size)
        return self.model(input_ids, token_type_ids, attention_mask).logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          labels=labels)

        if self.config.task == "multi":
            loss = cross_entropy(softmax(result.logits, dim=1), labels)
            self.train_acc(result.logits, labels)
            self.train_f1(result.logits, labels)
        
        else:
            loss = binary_cross_entropy_with_logits(result.logits, labels.float().unsqueeze(1))
            self.train_acc(result.logits, labels.unsqueeze(1))
            self.train_f1(result.logits, labels.unsqueeze(1))

        
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc, on_epoch=True)
        self.log('train_f1', self.train_f1, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        if self.config.task == "multi":
            loss = cross_entropy(softmax(result.logits, dim=1), labels)
            self.valid_acc(result.logits, labels)
            self.valid_f1(result.logits, labels)
        
        else:
            loss = binary_cross_entropy_with_logits(result.logits, labels.float().unsqueeze(1))
            self.valid_acc(result.logits, labels.unsqueeze(1))
            self.valid_f1(result.logits, labels.unsqueeze(1))

        
        self.log("val_loss", loss)
        self.log('val_acc', self.valid_acc, on_epoch=True)
        self.log('val_f1', self.valid_f1, on_epoch=True)

        
    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        if self.config.task == "multi":
            loss = cross_entropy(softmax(result.logits, dim=1), labels)
            self.test_acc(result.logits, labels)
            self.test_f1(result.logits, labels)
        
        else:
            loss = binary_cross_entropy_with_logits(result.logits, labels.float().unsqueeze(1))
            self.test_acc(result.logits, labels.unsqueeze(1))
            self.test_f1(result.logits, labels.unsqueeze(1))

        
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        
    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=1000,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optim], [scheduler]
