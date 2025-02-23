import json

# Input and output file paths
jsonl_file = "FINAL_finetuning_data_ques_only.jsonl"  # Replace with your file name
json_file = "FINAL_finetuning_data_ques_only.json"   # Output JSON file

# Read the JSONL file and convert to a list of dictionaries
with open(jsonl_file, "r", encoding="utf-8") as infile:
    data = [json.loads(line) for line in infile]

# Write to a JSON file
with open(json_file, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=4)

print(f"Converted {jsonl_file} to {json_file} successfully!")
