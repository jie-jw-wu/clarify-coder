import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import nltk
from nltk import FreqDist

# Dependencies required:
# pip install torch transformers numpy nltk

# Ensure that you have downloaded the NLTK punkt tokenizer
nltk.download('punkt')
# Couldn't find punkt_tab even after installing punkt, so manually installing it
nltk.download('punkt_tab')

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_perplexity(text, model, tokenizer, max_length=1024):
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    if tokens.size(1) > max_length:
        tokens = tokens[:, :max_length]  # truncate to the model's maximum input length (GPT has max token length of 1024)
    
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
    return torch.exp(loss).item()


def compute_entropy(text):
    words = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(words)
    total_words = len(words)
    
    # Calculate entropy
    probabilities = np.array(list(word_freq.values())) / total_words
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add small value to avoid log(0)
    
    return entropy

file_path = 'A1_finetuning/finetuning_data_ques_only.jsonl'
data = load_jsonl(file_path)

model_name = 'gpt2'  # can choose to use another model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

results = []

for entry in data:
    problem = entry['problem']
    answer = entry['answer']
    
    problem_perplexity = compute_perplexity(problem, model, tokenizer)
    answer_perplexity = compute_perplexity(answer, model, tokenizer)
    
    problem_entropy = compute_entropy(problem)
    answer_entropy = compute_entropy(answer)
    
    results.append({
        'problem': problem,
        'answer': answer,
        'problem_perplexity': problem_perplexity,
        'answer_perplexity': answer_perplexity,
        'problem_entropy': problem_entropy,
        'answer_entropy': answer_entropy
    })

# print results
for res in results:
    print(f"Problem: {res['problem']}")
    print(f"Answer: {res['answer']}")
    print(f"Problem Perplexity: {res['problem_perplexity']:.4f}, Problem Entropy: {res['problem_entropy']:.4f}")
    print(f"Answer Perplexity: {res['answer_perplexity']:.4f}, Answer Entropy: {res['answer_entropy']:.4f}\n")

# save results to a JSONL file
# output_file_path = 'results.jsonl'
# with open(output_file_path, 'w') as f:
#     for res in results:
#         f.write(json.dumps(res) + '\n')
