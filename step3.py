# Generating Clarifying Questions and Clarification Score using modified problem statements.
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

You are given an ambiguous coding problem description. Your task is to assess the level of ambiguity and provide a clarification score as well as ask necessary clarifying questions to resolve the ambiguity.

Definitions:

Ambiguity: A problem statement is ambiguous if it includes multiple valid interpretations or has unspecified details.
Clarification Score: A measure of how necessary it is to ask clarifying questions to complete the coding task. A score of 0 means no clarifying questions are needed, while a score of 1 means clarifying questions are absolutely necessary.

Ambiguous Coding Problem Description:
{question}

Output Format:

Clarifying Questions: Ask clarifying questions that would help remove the ambiguity from the problem statement.
Score: Provide a clarification score ranging from 0 to 1.

## COPY TILL HERE ##
'''

'''
PROMPT TEMPLATE TO GENERATE INCONSISTENT PROBLEM

## COPY FROM HERE ##

You are given an inconsistent coding problem description. Your task is to assess the level of inconsistency and provide a clarification score as well as ask necessary clarifying questions to resolve the inconsistency.

Definitions:

Inconsistency: A problem description becomes inconsistent if some statements in the description show conflict.
Clarification Score: A measure of how necessary it is to ask clarifying questions to complete the coding task. A score of 0 means no clarifying questions are needed, while a score of 1 means clarifying questions are absolutely necessary.

Inconsistent Coding Problem Description:
{question}

Output Format:

Clarifying Questions: Ask clarifying questions that would help remove the inconsistency from the problem statement.
Score: Provide a clarification score ranging from 0 to 1.

## COPY TILL HERE ##

'''

'''
PROMPT TEMPLATE TO GENERATE INCOMPLETE PROBLEM

## COPY FROM HERE ##

You are given an incomplete coding problem description. Your task is to assess the level of incompleteness and provide a clarification score as well as ask necessary clarifying questions to resolve the incompleteness.

Definitions:

Incompleteness: Absence of some of the key concepts and conditions that are crucial for solving the problem makes it incomplete.
Clarification Score: A measure of how necessary it is to ask clarifying questions to complete the coding task. A score of 0 means no clarifying questions are needed, while a score of 1 means clarifying questions are absolutely necessary.

Incomplete Coding Problem Description:
{question}

Output Format:

Clarifying Questions: Ask clarifying questions that would help remove the incompleteness from the problem statement.
Score: Provide a clarification score ranging from 0 to 1.

## COPY TILL HERE ##

'''

# PROMPT TEMPLATE (please change to get Ambiguous/Inconsistent/Incomplete description by copying the prompt from the options above)
TEMPLATE = """
You are given an ambiguous coding problem description. Your task is to assess the level of ambiguity and provide a clarification score as well as ask necessary clarifying questions to resolve the ambiguity.

Definitions:

Ambiguity: A problem statement is ambiguous if it includes multiple valid interpretations or has unspecified details.
Clarification Score: A measure of how necessary it is to ask clarifying questions to complete the coding task. A score of 0 means no clarifying questions are needed, while a score of 1 means clarifying questions are absolutely necessary.

Ambiguous Coding Problem Description:
{question}

Output Format:

Clarifying Questions: Ask clarifying questions that would help remove the ambiguity from the problem statement.
Score: Provide a clarification score ranging from 0 to 1.
"""

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

def load_questions(dir_path):
    
    formatted_data = []
    folder_paths = sorted(glob.glob(os.path.join(dir_path, '*')))
    
    for i, folder_path in enumerate(tqdm(folder_paths, desc="Processing Folders", unit="folder"), start=1):
        question_file_path = os.path.join(folder_path, "modified_question.txt")
        if os.path.isfile(question_file_path):
            with open(question_file_path, 'r') as file:
                question_text = file.read().strip()
                formatted_entry = TEMPLATE.format(question=question_text)
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
    args = parser.parse_args()

    print("Phase 1: Loading questions")
    formatted_data = load_questions(args.dir_path)

    print("Phase 2: Configuring generative model")
    model = configure_genai(args.api_key)

    print("Phase 3: Generating responses and saving to JSONL file")
    generate_responses(model, formatted_data, args.jsonl_file_path)
    # SAVING TO JSONL
    # This format stores each data entry as a separate JSON object on a new line, making it easy to read and process using HF datasets library.

    print(f"\nJSONL file with input-output pairs saved at: {args.jsonl_file_path}")

if __name__ == "__main__":
    main()