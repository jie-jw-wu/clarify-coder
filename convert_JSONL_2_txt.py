import os
import json

def process_jsonl_file(jsonl_file):
    with open(jsonl_file, 'r') as file:
        for index, line in enumerate(file):
            data = json.loads(line)
            output_value = data.get('output')
            
            if output_value:
                folder_name = str(index)
                os.makedirs(folder_name, exist_ok=True)
                
                output_file_path = os.path.join(folder_name, 'modified_question.txt')
                with open(output_file_path, 'w') as output_file:
                    output_file.write(output_value)

if __name__ == "__main__":
    jsonl_file = 'path_to_your_file.jsonl'
    process_jsonl_file(jsonl_file)