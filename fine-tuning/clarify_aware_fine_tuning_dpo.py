from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch


model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  
    load_in_4bit=True,
    device_map="auto"
)

model.config.use_cache = False  
model.gradient_checkpointing_enable()  

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

model = get_peft_model(model, peft_config)
dataset = load_dataset("json", data_files="dpo_data/dpo_finetune_data.jsonl")
dataset = dataset.map(lambda x: {"prompt": x["chosen"], "rejected": x["rejected"]})

training_args = TrainingArguments(
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
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
    train_dataset=dataset["train"] if "train" in dataset else dataset,
    tokenizer=tokenizer,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

dpo_trainer.train()

dpo_trainer.save_model("./dpo_finetuned_model")
tokenizer.save_pretrained("./dpo_finetuned_model")

print("----------------Fine-tuning complete!-------------------")