import math
import time

import nltk
import openai
import re
import os
import json
import subprocess
import argparse
import random
import string
from nltk.corpus import stopwords

import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# fine-tuning tutorial: 
## https://www.kaggle.com/code/lizhecheng/qlora-fine-tune-gpt-neox-20b
## https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=cg3fiQOvmI3Q
## https://colab.research.google.com/github/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb#scrollTo=_TAjrSWSe14q
## https://huggingface.co/blog/peft

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, help='Path to the model')
parser.add_argument('--finetune_method', type=str, default='lora', help='fine-tuning method')
parser.add_argument("-m","--model",help="LLM",type=str,required=True)
parser.add_argument('--use_int8', action='store_true', help='whether to use int8 quantization')
parser.add_argument('--use_fp16', action='store_true', help='whether to use fp16 precision')
parser.add_argument("--dataset_path",help="dataset_path",type=str,required=True)
parser.add_argument("--finetuned_model_path",help="finetuned_model_path",type=str,required=True)

args = parser.parse_args()

# TODO: Load the dataset
# https://github.com/huggingface/datasets/issues/824#issuecomment-758358089

#data = load_dataset("Abirate/english_quotes")

data = datasets.load_from_disk(args.dataset_path)
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

if args.use_int8:
    print("**********************************")
    print("**** Using 8-bit quantization ****")
    print("**********************************")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        cache_dir=HF_HOME,
        offload_folder=offload_folder, 
        local_files_only=True,     
    )
# if specified, use fp16 precision
elif args.use_fp16:
    print("**********************************")
    print("****** Using fp16 precision ******")
    print("**********************************")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=HF_HOME,
        offload_folder=offload_folder,     
        local_files_only=True,     
    )
# otherwise, use default precision
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        cache_dir=HF_HOME,
        offload_folder=offload_folder,   
        local_files_only=True,              
    )


# configure tokenizer
if (args.model.startswith('Meta-Llama')
    or args.model.startswith('deepseek')
    or args.model.startswith('CodeQwen')):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.seq_length,
        # Bug: A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
        # setting padding_side='left' doesn't fix the issue.
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=HF_HOME,
        offload_folder=offload_folder,
    )


### Post-processing on the model
# Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast the layer-norm in `float32` for stability. We also cast the output of the last layer in `float32` for the same reasons.
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=1,
)

if args.finetune_method == "lora":
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,                     # Rank of the low-rank matrices
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1,        # Dropout for LoRA layers
        target_modules=["query", "value"]  # Target modules to apply LoRA
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
else:
    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

# Train the model
trainer.train()

model.save_pretrained(args.finetuned_model_path)

# Inference
batch = tokenizer("Two things are infinite: ", return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))