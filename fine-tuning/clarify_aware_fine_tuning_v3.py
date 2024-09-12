
import argparse
import os
import sys
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import load_from_disk, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from sklearn.model_selection import train_test_split

# fine-tuning tutorial: 
## https://www.kaggle.com/code/lizhecheng/qlora-fine-tune-gpt-neox-20b
## https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=cg3fiQOvmI3Q
## https://colab.research.google.com/github/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb#scrollTo=_TAjrSWSe14q
## https://colab.research.google.com/drive/14xo6sj4dARk8lXZbOifHEn1f_70qNAwy?usp=sharing
## https://huggingface.co/blog/peft
## https://github.com/ragntune/code-llama-finetune/tree/main?tab=readme-ov-file
def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(name, param.device)
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def tokenize_function(samples):
    return tokenizer(samples['problem'])# + samples['answer'], 
    #padding="max_length", max_length=4000)  #, padding=True, truncation=True, max_length=128)

def tokenize_function2(samples):
    print(samples)
    
    # Return the concatenated text in a dict format
    return {'concatenated_text': concatenated_text}

def tokenize(samples):
    concatenated_text = samples['problem'] + samples['answer'] + samples['type']
    result = tokenizer(
        concatenated_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, help='Path to the model',required=True)
parser.add_argument('--finetune_method', type=str, default='lora', help='fine-tuning method')
parser.add_argument("-m","--model",help="LLM",type=str)
parser.add_argument('--use_int8', action='store_true', help='whether to use int8 quantization')
parser.add_argument('--use_fp16', action='store_true', help='whether to use fp16 precision')
parser.add_argument("--dataset_path",help="dataset_path",type=str,required=True)
parser.add_argument("--finetuned_model_path",help="finetuned_model_path",type=str,required=True)
parser.add_argument('--checkpoint', type=str, default="", help='checkpoint file')
parser.add_argument("--output_dir",type=str,required=True)

args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# TODO: Load the dataset
# https://github.com/huggingface/datasets/issues/824#issuecomment-758358089
# data = load_dataset("Abirate/english_quotes")
#data = load_dataset('json', data_files=args.dataset_path)
#data = data.map(tokenize_function2, batched=True, batch_size=8)
#data = data.map(tokenize_function2, batched=False)

HF_HOME = "/scratch/jie"
offload_folder = "offload_folder"


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
        torch_dtype=torch.float16,
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
        load_in_8bit=True,  
        local_files_only=True,          
        #cache_dir=HF_HOME,
        #offload_folder=offload_folder,      
        )

# configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    device_map='auto',
)

#tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# TODO: Load the dataset
# https://github.com/huggingface/datasets/issues/824#issuecomment-758358089
# data = load_dataset("Abirate/english_quotes")
data = load_dataset('json', data_files=args.dataset_path)
tokenized_data = data.map(tokenize)
#data = data.map(tokenize_function, batched=True, batch_size=8)

# Split the dataset into training and validation sets (80% train, 20% validate)
print(data)
print(tokenized_data)
train_val_split = tokenized_data['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Check the size of the datasets
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Inspect a sample from the train and validation datasets
print(train_dataset[0])
print(val_dataset[0])


model.train() # put model back into training mode
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

resume_from_checkpoint = args.checkpoint # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


print_trainable_parameters(model)

# Define the Trainer
batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = args.output_dir #"code-llama-fine-tuned-v1"

training_args = TrainingArguments(
        #per_device_train_batch_size=4,
        #gradient_accumulation_steps=4,
        per_device_train_batch_size=4,#per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=40,
        save_steps=40,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="none", # if use_wandb else "none",
        run_name=None, # if use_wandb else None,
        remove_unused_columns=True,
    )

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

model.save_pretrained(args.finetuned_model_path)

# Inference
# TODO: update eval
batch = tokenizer("Two things are infinite: ", return_tensors='pt').to('cuda') 

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))