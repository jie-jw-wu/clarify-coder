from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import argparse
import os
import sys


model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto"
)

model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj',
                    'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("dpo_data/dpo_finetune_data.jsonl")
dataset = dataset.rename_column({
    "chosen": "prompt",
    "rejected": "rejected"
})

training_args = TrainingArguments(
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir="./dpo_finetuned_model",
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
)


dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

dpo_trainer.train()

dpo_trainer.save_model("./dpo_finetuned_model")
tokenizer.save_pretrained("./dpo_finetuned_model")

print("----------------Fine-tuning complete!-------------------")