# BERT_single_sequence_cls
This template is for fine-tuning BERT model (Single Sequence Classification task)

## How to use
You can run the template by running this script.  
```bash
>>> sh start_train.sh
```
in `start_train.sh` file, you can fix the hyperparameter of model.  
```bash
python train.py --model_name  "type your own value" \
                --tokenizer_path "type your own value" \
                --train_data_path "type your own value" \
                --valid_data_path "type your own value" \
                --wandb_project "type your own value" \
                --batch_size 16 \
                --device 0 \
                --max_source_length 512 \
                --vocab_size 64100
                
```