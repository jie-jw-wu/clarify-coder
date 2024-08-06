import json
import os
import random
import argparse

def add_type_attribute(input_file_path, output_file_path, type_value):
    data = []
    with open(input_file_path, 'r') as infile:
        for line in infile:
            record = json.loads(line)
            # check if "type" attribute is present, if not, add it
            if 'type' not in record:
                record['type'] = type_value
            data.append(record)
    
    # create output file if it doesn't exist
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as outfile:
            pass

    # append the updated records to the output file
    with open(output_file_path, 'a') as outfile:
        for record in data:
            json.dump(record, outfile)
            outfile.write('\n')
    
    # shuffle output
    with open(output_file_path, 'r') as outfile:
        all_data = [json.loads(line) for line in outfile]
    
    random.shuffle(all_data)

    with open(output_file_path, 'w') as outfile:
        for record in all_data:
            json.dump(record, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add 'type' attribute to JSONL data and shuffle.")
    parser.add_argument('input_file', help='Path to the input JSONL file.')
    parser.add_argument('type_value', help='Value for the "type" attribute.')
    args = parser.parse_args()

    # Define the output file path
    output_file_path = "finetuning_data.json"
    
    # Call the function to add the "type" attribute and shuffle data
    add_type_attribute(args.input_file, output_file_path, args.type_value)
