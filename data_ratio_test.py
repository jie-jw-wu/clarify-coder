import json
import random
from datasets import Dataset
from transformers import AutoTokenizer


def load_jsonl(file_path):
    """
        Loads a .jsonl file into a list of dictionaries
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def combine_datasets(questions_data, everything_data, ratio1=0.5, ratio2=0.5):
    """
        Combines two datasets (standard and synthetic data) using given ratios
    """
    num_items1 = int(len(questions_data) * ratio1)
    num_items2 = int(len(everything_data) * ratio2)

    sampled_questions = random.sample(questions_data, num_items1)
    sampled_answers = random.sample(everything_data, num_items2)

    combined_data = sampled_questions + sampled_answers
    random.shuffle(combined_data)
    return combined_data

def format_combined_data(combined_data):
    """
        Format the combined data
    """
    formatted_data = []
    for entry in combined_data:
        if "problem" in entry and "answer" in entry:
            formatted_data.append({
                "prompt": entry["problem"],
                "response": entry["answer"]
            })
    return formatted_data


questions_data = load_jsonl(
    '/Users/arhaankhaku/Documents/Development/Projects/clarify-aware-coder/FINAL_finetuning_data_ques_only.jsonl')
everything_data = load_jsonl(
    '/Users/arhaankhaku/Documents/Development/Projects/clarify-aware-coder/FINAL_finetuning_everything.jsonl')

combined_data = combine_datasets(
    questions_data, everything_data, ratio1=0.7, ratio2=0.3)
formatted_data = format_combined_data(combined_data)

dataset = Dataset.from_dict({
    'prompt': [entry['prompt'] for entry in formatted_data],
    'response': [entry['response'] for entry in formatted_data]
})

dataset.save_to_disk('./combined_finetuning_data')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token' : '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)


