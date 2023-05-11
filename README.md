# LLM-Train-Template
Repository that contains LLM train template that makes LLM training easier.  
Templates will be produced in two versions using Pytorch-Lightning and HuggingFace Trainer, respectively

Now, only Pytorch-Lightning version is available. 🚧

Template can be compatible with Weights and Biases, and DeepSpeed.  

<br>
<div align=center>
    <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=flat-square&logo=Python&logoColor=white"/>
  <img alt="PyTorch" src ="https://img.shields.io/badge/PyTorch-EE4C2C.svg?&style=flat-square&logo=PyTorch&logoColor=white"/>
  <img alt="PyTorch Lightning" src ="https://img.shields.io/badge/PyTorch Lightning-792EE5.svg?&style=flat-square&logo=PyTorch&logoColor=white"/>
  <img alt="weights and biases" src ="https://img.shields.io/badge/weights and biases-FFBE00.svg?&style=flat-square&logo=Plotly&logoColor=white"/>
  <img alt="Docker" src ="https://img.shields.io/badge/Docker-2496ED.svg?&style=flat-square&logo=Docker&logoColor=white"/>
</div>
<br>


## How to use
Basically, These templates are focused on CLI environment.  
so you can change hyperparameter of model by argparse and shell script.  
For each template, there is an `start_train.sh`.  
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
 
🚨 **Also, some modifications in template are needed to fit each environment and task** 🚨   

## Environment
This template is tested on [this docker image](https://hub.docker.com/r/pytorchlightning/pytorch_lightning).  

However, for users who do not use docker, the version of the core libraries is specified below (python 3.9.13)
```text
deepspeed==0.6.4
huggingface-hub==0.14.1
numpy==1.23.0
pandas==1.4.3
sentencepiece==0.1.99
torch==1.10.2+cu111
torcheval==0.0.6
torchmetrics==0.9.2
torchtext==0.11.2
torchtnt==0.1.0
torchvision==0.11.3+cu111
transformers==4.28.1
pytorch-lightning==1.7.0
```


## Models
(Now only pytorch lightning version is available)
- BERT 
- T5
