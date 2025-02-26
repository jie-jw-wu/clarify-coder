import argparse
import os
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# model_utils.py


def load_model(args):
    HF_HOME = args.hf_dir
    offload_folder = "offload_folder"
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    print("Loading model...")
    if args.do_save_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path      
        )
    elif args.use_int8:
        print("**********************************")
        print("**** Using 8-bit quantization ****")
        print("**********************************")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=HF_HOME,
            offload_folder=offload_folder,     
        )
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
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            cache_dir=HF_HOME,
            offload_folder=offload_folder,            
        )

    if 'finetuned' in args.model:
        model = PeftModel.from_pretrained(model, args.finetuned_model_path)

    print('model device: ', model.device)
    return model

def load_tokenizer(args):
    HF_HOME = args.hf_dir
    offload_folder = "offload_folder"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.seq_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=HF_HOME,
        offload_folder=offload_folder,
    )
    return tokenizer

def test_model(tokenizer, model, user_input, max_length):
    timea = time.time()
    input_ids = tokenizer(user_input, return_tensors="pt")["input_ids"].to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_length)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    print('!!!!!!!!!!')
    print(filling)
    print('!!!!!!!!!!')
    print("timea = time.time()", -timea + time.time())

def save_model(model, tokenizer, saved_model_path):
    tokenizer.save_pretrained(saved_model_path)
    model.save_pretrained(saved_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')

    args = parser.parse_args()

    #model = load_model(args)
    #tokenizer = load_tokenizer(args)

    #test_model(tokenizer, model, args.user_input, args.seq_length)

    #evaluate_model(args.model_path, args.dataset_path, args.output_dir)
    data = load_dataset('json', data_files=args.dataset_path)

    model = load_model(args)
    tokenizer = load_tokenizer(args)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple accuracy metric for demonstration
        accuracy = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=data['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")