import torch
import numpy as np
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config, get_linear_schedule_with_warmup
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, softmax
from torcheval.metrics.text import Perplexity
from torchmetrics import BLEUScore

class T5ModelForPreTraining(pl.LightningModule):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()        
        self.T5Config = T5Config(
            vocab_size= config.vocab_size,
            d_ff= 2048,
            d_kv= 64,
            d_model= 768,
            decoder_start_token_id= 0,
            dropout_rate= 0.1,
            eos_token_id= 1,
            feed_forward_proj= "gated-gelu",
            initializer_factor= 1.0,
            is_encoder_decoder= True,
            layer_norm_epsilon= 1e-06,
            n_positions= 512,
            num_decoder_layers= 12,
            num_heads= 12,
            num_layers= 12,
            pad_token_id= 0,
            relative_attention_num_buckets= 32,
        )
        
        if config.model_name == "None":
            self.model = T5ForConditionalGeneration(self.T5Config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(config.model_name, vocab_size=config.vocab_size, ignore_mismatched_sizes=True)
        self.save_hyperparameters()
        self.config = config
        self.bleu = BLEUScore()
        self.tokenizer = tokenizer

    def forward(self, batch):
        # (batch_size, sequence_length, config.vocab_size)
        input_ids, attention_mask, labels = batch
        return self.model(
                    input_ids= input_ids, 
                    attention_mask= attention_mask).logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        loss = result.loss

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        loss = result.loss
        
        self.log("val_loss", loss, sync_dist=True)


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        result = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        
        loss = result.loss
        labels_hat = torch.argmax(result.logits, dim=1)
        labels_hat = self.tokenizer.decode(labels_hat[0], skip_special_tokens=True)
        labels[[labels == -100]] = 0
        label = self.tokenizer.decode(labels[0], skip_special_tokens=True)
        bleu_score = self.bleu([labels_hat], [[label]])

        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_bleu", bleu_score, sync_dist=True)
        
    
    def generate(self, input_ids):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        
        return self.model.generate(input_ids=input_ids, max_length=512)
        
        
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
        # optim = FusedAdam(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optim], [scheduler]