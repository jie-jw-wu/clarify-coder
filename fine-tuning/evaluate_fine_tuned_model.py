import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--tokenize_version', type=int, choices=[1, 2, 3, 4], required=True, help='Select which tokenize function to use: 1, 2, 3, or 4')
    args = parser.parse_args()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Define the tokenize function based on the version
    def tokenize_v1(samples):
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

    def tokenize_v2(samples):
        concatenated_text = samples['problem'] + samples['answer']
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        problem_tokens = tokenizer(samples['problem'], truncation=True, max_length=512, padding=False, return_tensors=None)["input_ids"]
        answer_tokens = tokenizer(samples['answer'], truncation=True, max_length=512, padding=False, return_tensors=None)["input_ids"]
        answer_start_idx = len(problem_tokens)
        labels = [-100] * len(result["input_ids"])
        labels[answer_start_idx:answer_start_idx + len(answer_tokens)] = result["input_ids"][answer_start_idx:answer_start_idx + len(answer_tokens)]
        result["labels"] = labels
        return result

    def tokenize_v3(samples):
        concatenated_text = samples['problem'] + samples['answer'] + samples['type']
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_v4(samples):
        QPROMPT = "You are an expert software developer who writes high quality code. With below information, please either generate Python3 code (Respond directly with code only with markdown), or ask clarifying questions:\n"
        if samples['type'] == "Original":
            APROMPT = "This is a clear problem requiring no clarifications. Let's generate the required Python3 code directly in markdown."
        else:
            APROMPT = "I have a few clarifying questions. Please respond with the necessary details so I can assist further."
        concatenated_text = f"{QPROMPT} {samples['problem']}" + f"{APROMPT} {samples['answer']}"
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    if args.tokenize_version == 1:
        tokenize_fn = tokenize_v1
    elif args.tokenize_version == 2:
        tokenize_fn = tokenize_v2
    elif args.tokenize_version == 3:
        tokenize_fn = tokenize_v3
    elif args.tokenize_version == 4:
        tokenize_fn = tokenize_v4

    # Load the dataset
    data = load_dataset('json', data_files=args.dataset_path)
    tokenized_data = data.map(tokenize_fn)
    val_dataset = tokenized_data['train'].train_test_split(test_size=0.2, seed=42)['test']

    # Define the Trainer
    training_args = TrainingArguments(
        per_device_eval_batch_size=16,
        output_dir='./results',
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
