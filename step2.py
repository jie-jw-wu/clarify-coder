# Code that converts Original Problem Description to Modified (Ambiguous/Inconsistent/Incomplete) description
import os
import glob
import json
import time
import argparse
from tqdm import tqdm
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

'''
PROMPT TEMPLATE TO GENERATE AMBIGUOUS PROBLEM

## COPY FROM HERE ##

Based on the knowledge that ambiguous problem descriptions can be created by introducing multiple valid interpretations or unspecified details, think step-by-step to rewrite the given coding problem description and make it ambiguous. Only output the modified problem description itself.

Original Problem Description:
{question}

Ambiguous Problem:

## COPY TILL HERE ##
'''

'''
PROMPT TEMPLATE TO GENERATE INCONSISTENT PROBLEM

## COPY FROM HERE ##

Based on the knowledge that a problem becomes inconsistent if some statements in the description show conflict, think step-by-step to rewrite the given coding problem description and make it inconsistent. Only output the modified problem description itself.

Original Problem Description:
{question}

Inconsistent Problem:

## COPY TILL HERE ##

'''

'''
PROMPT TEMPLATE TO GENERATE INCOMPLETE PROBLEM

## COPY FROM HERE ##

Based on the knowledge that removing some of the key concepts and conditions that are crucial for solving the problem makes it incomplete, think step-by-step to rewrite the given coding problem description to make it incomplete. Only output the modified problem description itself.

Original Problem Description:
{question}

Incomplete Problem:

## COPY TILL HERE ##

'''

# Map prompt types to their respective templates
PROMPT_TEMPLATES = {
    "ambiguous": """
Based on the knowledge that ambiguous problem descriptions can be created by introducing multiple valid interpretations or unspecified details, think step-by-step to rewrite the given coding problem description and make it ambiguous. Only output the modified problem description itself.

Original Problem Description:
{question}

Ambiguous Problem:
""",
    "inconsistent": """
Based on the knowledge that a problem becomes inconsistent if some statements in the description show conflict, think step-by-step to rewrite the given coding problem description and make it inconsistent. Only output the modified problem description itself.

Original Problem Description:
{question}

Inconsistent Problem:
""",
    "incomplete": """
Based on the knowledge that removing some of the key concepts and conditions that are crucial for solving the problem makes it incomplete, think step-by-step to rewrite the given coding problem description to make it incomplete. Only output the modified problem description itself.

Original Problem Description:
{question}

Incomplete Problem:
"""
}
def configure_genai(api_key):
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

def load_questions(dir_path, template):
    
    formatted_data = []
    folder_paths = sorted(glob.glob(os.path.join(dir_path, '*')))
    
    for i, folder_path in enumerate(tqdm(folder_paths, desc="Processing Folders", unit="folder"), start=1):
        question_file_path = os.path.join(folder_path, "question.txt")
        if os.path.isfile(question_file_path):
            with open(question_file_path, 'r') as file:
                question_text = file.read().strip()
                formatted_entry = template.format(question=question_text)
                formatted_data.append(formatted_entry)
    
    return formatted_data

def generate_responses(model, formatted_data, jsonl_file_path):
    with open(jsonl_file_path, 'a') as jsonl_file:
        for i, prompt in enumerate(tqdm(formatted_data, desc="Generating Responses", unit="entry"), start=1):
            try:
                response = model.generate_content(prompt)
                input_output_pair = {'input': prompt, 'output': response.text}
                jsonl_file.write(json.dumps(input_output_pair) + '\n')
            except ResourceExhausted:
                print(f"ResourceExhausted error occurred for prompt {i}. Retrying after a delay...")
                time.sleep(60)
                try:
                    response = model.generate_content(prompt)
                    input_output_pair = {'input': prompt, 'output': response.text}
                    jsonl_file.write(json.dumps(input_output_pair) + '\n')
                except Exception as e:
                    print(f"Error occurred for prompt {i} even after retrying: {e}")
                    input_output_pair = {'input': prompt, 'output': str(response.prompt_feedback)}
                    jsonl_file.write(json.dumps(input_output_pair) + '\n')
            except ValueError as e:
                print(f"Error occurred for prompt {i}: {e}")
                input_output_pair = {'input': prompt, 'output': str(response.prompt_feedback)}
                jsonl_file.write(json.dumps(input_output_pair) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate appropriate response given prompt using Google Generative AI.")
    parser.add_argument('--api_key', type=str, required=True, help="API key for Google Generative AI.")
    parser.add_argument('--dir_path', type=str, help="Directory containing folders with coding problems.")
    parser.add_argument('--jsonl_file_path', type=str, help="Path to save the output JSONL file.")
    parser.add_argument('--type', type=str, choices=['ambiguous', 'inconsistent', 'incomplete'], required=True, help="Type of problem description to generate (ambiguous, inconsistent, incomplete).")
    args = parser.parse_args()

    # Select the appropriate template based on the chosen type
    template = PROMPT_TEMPLATES[args.type]

    # Ensure the directory for the JSONL file path exists
    jsonl_directory = os.path.dirname(args.jsonl_file_path)
    if jsonl_directory and not os.path.exists(jsonl_directory):
        os.makedirs(jsonl_directory)

    print("Phase 1: Loading questions")
    formatted_data = load_questions(args.dir_path, template)

    print("Phase 2: Configuring generative model")
    model = configure_genai(args.api_key)

    print("Phase 3: Generating responses and saving to JSONL file")
    generate_responses(model, formatted_data, args.jsonl_file_path)
    # SAVING TO JSONL
    # This format stores each data entry as a separate JSON object on a new line, making it easy to read and process using HF datasets library.

    print(f"\nJSONL file with input-output pairs saved at: {args.jsonl_file_path}")

if __name__ == "__main__":
    main()