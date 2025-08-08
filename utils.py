import json
import os
import random
import yaml
import subprocess
import sys
from pathlib import Path
import google.generativeai as genai


def convert_jsonl_to_json(jsonl_file, json_file):
    """
    Convert a JSONL file to a JSON file.
    
    Args:
        jsonl_file (str): Path to the input JSONL file
        json_file (str): Path to the output JSON file
    """
    # Read the JSONL file and convert to a list of dictionaries
    with open(jsonl_file, "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile]

    # Write to a JSON file
    with open(json_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Converted {jsonl_file} to {json_file} successfully!")


def process_jsonl_file(jsonl_file, output_dir):
    """
    Process a JSONL file and save outputs to separate directories.
    
    Args:
        jsonl_file (str): Path to the JSONL file
        output_dir (str): Directory to save the output folders and files
    """
    with open(jsonl_file, 'r', encoding="utf-8") as file:
        for index, line in enumerate(file):
            data = json.loads(line)
            output_value = data.get('output')
            
            if output_value:
                folder_name = str(index)
                folder_path = os.path.join(output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                
                output_file_path = os.path.join(folder_path, 'modified_question.txt')
                with open(output_file_path, 'w', encoding="utf-8") as output_file:
                    output_file.write(output_value)


def oversample_data(data, target_count):
    """
    Oversample data to reach target count using random choices with replacement.
    
    Args:
        data (list): Source data to sample from
        target_count (int): Target number of samples
        
    Returns:
        list: Sampled data
    """
    return random.choices(data, k=target_count) if target_count > len(data) else random.sample(data, target_count)


def downsample_data(data, target_count):
    """
    Downsample data to target count using random sampling without replacement.
    
    Args:
        data (list): Source data to sample from
        target_count (int): Target number of samples
        
    Returns:
        list: Sampled data
    """
    return random.sample(data, min(target_count, len(data)))


def generate_original_type(file_path, output_dir, output_filename="original.jsonl"):
    """
    Filter and save only items with type "Original" from a JSONL file.
    
    Args:
        file_path (str): Path to input JSONL file
        output_dir (str): Output directory
        output_filename (str): Output filename (default: "original.jsonl")
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    original_data = [item for item in data if item.get('type') == "Original"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / output_filename

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in original_data:
            f.write(json.dumps(item) + '\n')

    print(f"Filtered data saved to: {output_file}")


def split_and_save_jsonl(file_path, output_dir, ratio, sampling_mode):
    """
    Split JSONL file into original and other data according to ratio and sampling mode.
    
    Args:
        file_path (str): Path to input JSONL file
        output_dir (str): Output directory for split files
        ratio (tuple): Ratio of (original_percent, other_percent)
        sampling_mode (str): Either "oversample" or "downsample"
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # extracts original data
    original_data = [item for item in data if item.get('type') == "Original"]
    # extracts everything except original data
    other_data = [item for item in data if item.get('type') != "Original"]

    # total number of samples available to us
    total_desired_samples = min(len(original_data) / (ratio[0] / 100), len(other_data) / (ratio[1] / 100))
    total_desired_samples = int(total_desired_samples)

    # number of samples for original and other according to the ratio
    original_count = int(ratio[0] / 100 * total_desired_samples)
    other_count = int(ratio[1] / 100 * total_desired_samples)

    # Apply sampling based on the mode
    if sampling_mode == "oversample":
        selected_original = oversample_data(original_data, original_count)
        selected_other = oversample_data(other_data, other_count)
    elif sampling_mode == "downsample":
        selected_original = downsample_data(original_data, original_count)
        selected_other = downsample_data(other_data, other_count)
    else:
        raise ValueError("Invalid sampling_mode. Choose 'oversample' or 'downsample'.")

    # combines the original data and the other data and then shuffles them
    combined_data = selected_original + selected_other
    random.shuffle(combined_data)

    # Makes the output directory where the .jsonl files will be stored
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"split_{ratio[0]}_{ratio[1]}_{sampling_mode}.jsonl"
    with open(output_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"{sampling_mode} with {ratio[0]}% original and {ratio[1]}% other")
    print(f"Original Samples: {len(selected_original)}/{len(combined_data)}")
    print(f"Other Samples: {len(selected_other)}/{len(combined_data)}")
    print(f"Saved and split data to {output_file}")


def generate_worse_answer(correct_answer, type, model):
    """
    Generate a worse answer using AI model based on the type.
    
    Args:
        correct_answer (str): The correct answer to generate a worse version of
        type (str): Type of the data ("Original" or other)
        model: The AI model to use for generation
        
    Returns:
        str: Generated worse answer
    """
    if type == "Original":
        prompt = (
            "Given the following correct code, generate poorly formulated clarifying questions "
            "that are vague, irrelevant, misleading, or do not help resolve ambiguities:\n\n"
            + correct_answer
        )
    else:
        prompt = (
            "Given the following set of clarifying questions, generate an incorrect version of the code "
            "by introducing logical or syntactical errors while maintaining its general structure:\n\n"
            + correct_answer
        )
    response = model.generate_content(prompt)
    return response.text if response.text else "Error generating incorrect code"


def format_dpo_data(input_file, output_file, gemini_api_key, max_entries=5):
    """
    Format data for DPO (Direct Preference Optimization) training.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        gemini_api_key (str): API key for Gemini
        max_entries (int): Maximum number of entries to process (default: 5)
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    formatted_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_entries:
                break
            data = json.loads(line)

            if "problem" in data and "answer" in data:
                prompt = f"Problem: {data['problem']}\n\nSolution:"
                chosen = data["answer"]
                rejected = generate_worse_answer(chosen, data['type'], model)
                data_type = data['type']

                formatted_entry = {
                    "type": data_type,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
                formatted_data.append(formatted_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Formatted data saved to {output_file} with {len(formatted_data)} entries.")


def run_separate_jsonl_script():
    """
    Run the main logic from seperate_JSONL.py
    """
    input_file = "FINAL_finetuning_everything.jsonl"
    output_dir_splits = "output_splits"
    output_dir_original = "output_original"

    # Change desired ratio to what you want.
    desired_ratio = (40, 60)

    # Function call
    split_and_save_jsonl(input_file, output_dir_splits, desired_ratio, "oversample")
    generate_original_type(input_file, output_dir_original)


def run_dpo_data_format_script():
    """
    Run the main logic from dpo_data_format.py
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    input_file = "output_splits/split_20_80_downsample.jsonl"
    output_folder = "dpo_data"
    output_file = os.path.join(output_folder, "dpo_finetune_data.jsonl")

    # Use the utility function to format DPO data
    format_dpo_data(input_file, output_file, gemini_api_key, max_entries=5)


def run_convert_jsonl_to_json_script():
    """
    Run the main logic from convert_JSONL_2_JSON.py
    """
    # Input and output file paths
    jsonl_file = "FINAL_finetuning_data_ques_only.jsonl"  # Replace with your file name
    json_file = "FINAL_finetuning_data_ques_only.json"   # Output JSON file

    # Use the utility function to convert JSONL to JSON
    convert_jsonl_to_json(jsonl_file, json_file)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config