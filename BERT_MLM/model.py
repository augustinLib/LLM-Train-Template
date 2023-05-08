import torch
import numpy as np
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertConfig, get_linear_schedule_with_warmup
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, softmax


class TCRBertModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.bert_config = BertConfig(
            attention_probs_dropout_prob=0.1,
            classifier_dropout=None,
            gradient_checkpointing=False,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            initializer_range=0.02,
            intermediate_size=1536,
            layer_norm_eps=1e-12,
            max_position_embeddings=64,
            model_type="bert",
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            position_embedding_type="absolute",
            type_vocab_size=2,
            use_cache=True,
            vocab_size=25)
        
        self.model = BertForMaskedLM(self.bert_config)
        self.save_hyperparameters()
        self.config = config
        
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        # (batch_size, sequence_length, config.vocab_size)
        
        return self.model(input_ids, token_type_ids, attention_mask).logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        token_type_ids = torch.ones_like(batch["input_ids"])
        attention_mask = torch.where(input_ids[0] != 0, 1, 0)
        labels = batch["labels"]
        
        result = self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        loss = result.loss

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        token_type_ids = torch.ones_like(batch["input_ids"])
        attention_mask = torch.where(input_ids[0] != 0, 1, 0)
        labels = batch["labels"]
        result = self.model(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        loss = result.loss

        self.log("val_loss", loss)
        
        
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
            num_warmup_steps=10000,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optim], [scheduler]
