# T5_replace_corrupted_span
This template is for training T5 model with downstream text-to-text framework.  
Text-to-Text framework is proposed in [original T5 paper](https://arxiv.org/abs/1910.10683).  

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
                --vocab_size 64100 \
                --target_col \
                --source_col  \
                --prefix \
                
```