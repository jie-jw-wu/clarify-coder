import argparse
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import PeftModel


def get_ask_question_rate(response):
    if len(response_2_code(response)) == 0:
        return 1
    else:
        return 0

def response_2_code(response):
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[0] # code[-1] is the last triple code snippet
    else:
        return ''

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    comm_rate = get_ask_question_rate(pred.predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'comm_rate': comm_rate,
    }

# python in_dist_eval.py --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct  --finetuned_model /project/def-fard/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-02212025 --dataset_path /project/def-fard/jie/finetuning_data/FINAL_finetuning_data_ques_only.json  --tokenize_version 4
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the original model')
    parser.add_argument('--finetuned_model_path', type=str, required=True, help='Path to the finetuned model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--tokenize_version', type=int, choices=[1, 2, 3, 4], required=True, help='Select which tokenize function to use: 1, 2, 3, or 4')
    parser.add_argument('--output_dir', type=str, default='output-dir', help='Directory to save the output results')
    args = parser.parse_args()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, args.finetuned_model_path)

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


    # Output the results of the trained model for val_dataset in a new file
    output_file = os.path.join(args.output_dir, "validation_results.txt")
    with open(output_file, "w") as f:
        for i, sample in enumerate(val_dataset):
            inputs = tokenizer(sample["problem"], return_tensors='pt').to('cuda')
            with torch.cuda.amp.autocast():
                output_tokens = model.generate(**inputs, max_new_tokens=2000)
            output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            f.write(f"Sample {i}:\n")
            f.write(f"Problem: {sample['problem']}\n")
            f.write(f"Generated Answer: {output_text}\n")
            f.write(f"Actual Answer: {sample['answer']}\n\n")


    # Define the Trainer
    #training_args = TrainingArguments(per_device_eval_batch_size=16,output_dir='./results',logging_dir='./logs')

    #trainer = Trainer(model=model,args=training_args,eval_dataset=val_dataset,compute_metrics=compute_metrics)

    # Evaluate the model
    #results = trainer.evaluate()
    #print(results)

if __name__ == "__main__":
    main()