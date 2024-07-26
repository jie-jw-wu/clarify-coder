import os
import json

def combine_jsonl_files(input_folder, output_file):
    """
    Combines multiple JSONL files from a folder into one JSONL file.

    Parameters:
    input_folder (str): The folder containing the JSONL files to be combined.
    output_file (str): The output JSONL file.
    """
    try:
        with open(output_file, 'w') as outfile:
            for filename in os.listdir(input_folder):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(input_folder, filename)
                    with open(file_path, 'r') as infile:
                        for line in infile:
                            outfile.write(line)
        print(f"Combined JSONL files into {output_file}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Define the input folder and output file
    input_folder = "modified_problems"  # Change this to your input folder path
    output_file = "finetuning_data.jsonl"       # Change this to your desired output file path

    combine_jsonl_files(input_folder, output_file)
