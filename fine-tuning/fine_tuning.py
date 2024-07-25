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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, help='Path to the model')
parser.add_argument('--finetune_method', type=str, default='lora', help='fine-tuning method')
parser.add_argument("-m","--model",help="LLM",type=str,required=True)
parser.add_argument('--use_int8', action='store_true', help='whether to use int8 quantization')
parser.add_argument('--use_fp16', action='store_true', help='whether to use fp16 precision')

# TODO: Load the dataset
dataset = load_dataset("code_x_glue_ct_code_to_text")

# TODO: Assuming the dataset has a 'train' and 'validation' split
train_dataset = dataset['train']
validation_dataset = dataset['validation']

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

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['source'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# Set the format of the datasets to return PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

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
    model_lora = get_peft_model(model, lora_config)

    # Define the Trainer
    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
else:
    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
