import json
import random
from pathlib import Path

"""
    Splits .jsonl file into original data and other data and then combines them according to ratio
"""


def split_and_save_jsonl(file_path, output_dir, ratio):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # extracts original data
    original_data = [item for item in data if item.get('type') == "Original"]
    # extracts everything except original data
    other_data = [item for item in data if item.get('type') != "Original"]

    # total number of samples available to us
    max_original = len(original_data)
    max_other = len(other_data)
    total_samples = max_original + max_other

    # number of samples for original and other according to the ratio
    original_count = int(ratio[0]/100 * total_samples)
    other_count = int((ratio[1]/100) * total_samples)

    # If we need more original data than what we currently have:
    if original_count > max_original:
        other_count = min(max_other, other_count + (original_count - max_original))
        original_count = max_original
    # If we need more other data than what we currently have:
    if other_count > max_other:
        original_count = min(max_original, original_count + (other_count - max_other))
        other_count = max_other

    selected_original = random.sample(
        original_data, min(original_count, len(original_data)))
    selected_other = random.sample(
        other_data, min(other_count, len(other_data)))

    # combines the original data and the other data and then shuffles them
    combined_data = selected_original + selected_other
    random.shuffle(combined_data)

    # Makes the output directory where the .jsonl files will be stored
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f"split_{ratio[0]}_{ratio[1]}.jsonl"
    with open(output_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved and splitted data to {output_file}")


input_file = "FINAL_finetuning_everything.jsonl"
output_dir = "output_splits"
# Change desired ratio to what you want.
desired_ratio = (20, 80)
# Function call
split_and_save_jsonl(input_file, output_dir, desired_ratio)
