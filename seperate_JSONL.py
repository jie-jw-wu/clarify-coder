import json
import random
from pathlib import Path


def split_and_save_jsonl(file_path, output_dir, ratio):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    original_data = [item for item in data if item.get('type') == "Original"]
    other_data = [item for item in data if item.get('type') != "Original"]

    total_samples = len(original_data) + len(other_data)
    original_count = int(ratio[0]/100 * total_samples)
    other_count = int((ratio[1]/100) * total_samples)
    selected_original = random.sample(original_data, min(original_count, len(original_data)))
    selected_other = random.sample(other_data, min(other_count, len(original_data)))

    combined_data = selected_original + selected_other
    random.shuffle(combined_data)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f"split_{ratio[0]}_{ratio[1]}.jsonl"
    with open(output_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved and splitted data to {output_file}")


input_file = "FINAL_finetuning_everything.jsonl"
output_dir = "output_splits"
# Change desired ratio to what you want. 
desired_ratio = (20,80)

split_and_save_jsonl(input_file, output_dir, desired_ratio)
