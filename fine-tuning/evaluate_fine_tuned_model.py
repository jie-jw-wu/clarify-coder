import argparse
import os
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

def evaluate_model(model_path, dataset_path, output_dir):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the dataset
    data = load_dataset('json', data_files=dataset_path)
    
    # Tokenize the dataset
    def tokenize_function(samples):
        concatenated_text = samples['problem'] + samples['answer']
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_data = data.map(tokenize_function)
    val_dataset = tokenized_data['train'].train_test_split(test_size=0.2, seed=42)['test']

    # Define the Trainer
    training_args = TrainingArguments(
        per_device_eval_batch_size=8,
        output_dir=output_dir,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_model(args.model_path, args.dataset_path, args.output_dir)
