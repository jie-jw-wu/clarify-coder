import os
import json
import argparse

def process_jsonl_file(jsonl_file, output_dir):
    with open(jsonl_file, 'r') as file:
        for index, line in enumerate(file):
            data = json.loads(line)
            output_value = data.get('output')
            
            if output_value:
                folder_name = str(index)
                folder_path = os.path.join(output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                
                output_file_path = os.path.join(folder_path, 'modified_question.txt')
                with open(output_file_path, 'w') as output_file:
                    output_file.write(output_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file and save outputs to separate directories.")
    parser.add_argument('jsonl_file', type=str, help="Path to the JSONL file.")
    parser.add_argument('output_dir', type=str, help="Directory to save the output folders and files.")
    args = parser.parse_args()
    process_jsonl_file(args.jsonl_file, args.output_dir)
    print("converted jsonl file to folders and txt files")