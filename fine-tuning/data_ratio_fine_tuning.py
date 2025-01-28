import argparse
import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

def load_finetuning_data(dataset_path):
    """
    Load dataset in `.arrow` format.
    """
    dataset = load_dataset("arrow", data_files={"train": dataset_path})["train"]
    print(f"Loaded dataset with {len(dataset)} samples.")
    return dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}"
    )

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
parser.add_argument('--finetune_method', type=str, default='lora', help='Fine-tuning method')
parser.add_argument('--use_int8', action='store_true', help='Whether to use int8 quantization')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 precision')
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to save the fine-tuned model")
parser.add_argument("--tokenize_version", type=int, default=1, help="Tokenization version to use (1-4)")
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load model based on quantization or precision options
if args.use_int8:
    print("Using 8-bit quantization")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto")
elif args.use_fp16:
    print("Using fp16 precision")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")

# Configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_finetuning_data(args.dataset_path)
print("HERE ARE THE COLUMN NAMES\n", dataset.column_names)
# Tokenization logic
def tokenize_function(samples):
    if args.tokenize_version == 1:
        return tokenizer(samples['prompt'] + samples['response'], padding="max_length", truncation=True)
    elif args.tokenize_version == 2:
        return tokenizer(samples['response'], padding="max_length", truncation=True)
    elif args.tokenize_version == 3:
        return tokenizer(samples['prompt'] + samples['response'] + samples['type'], padding="max_length", truncation=True)
    elif args.tokenize_version == 4:
        if samples['type'] == "Original":
            return tokenizer("The problem is clear. Please write Python code for: " + samples['prompt'], padding="max_length", truncation=True)
        else:
            return tokenizer("The problem is not clear. Please ask clarifying questions for: " + samples['prompt'], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_data = dataset.map(tokenize_function, batched=True)
train_val_split = tokenized_data.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Apply LoRA if specified
if args.finetune_method == "lora":
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

# Define the Trainer
trainer = Trainer(
    model=model, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=args.use_fp16,
        logging_steps=10,
        output_dir=args.finetuned_model_path,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(args.finetuned_model_path)
tokenizer.save_pretrained(args.finetuned_model_path)

print(f"Model saved to {args.finetuned_model_path}")
