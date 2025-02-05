import json
import random
from pathlib import Path

"""
    Splits .jsonl file into original data and other data and then combines them according to ratio
"""
def oversample_data(data, target_count):
    return random.choices(data, k=target_count) if target_count > len(data) else random.sample(data, target_count)

def downsample_data(data, target_count):
    return random.sample(data, min(target_count, len(data)))

def split_and_save_jsonl(file_path, output_dir, ratio, sampling_mode):
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

    print(f"Saved and splitted data to {output_file}")


input_file = "FINAL_finetuning_everything.jsonl"
output_dir = "output_splits"
# Change desired ratio to what you want.
desired_ratio = (40, 60)
# Function call
split_and_save_jsonl(input_file, output_dir, desired_ratio, "oversample")
split_and_save_jsonl(input_file, output_dir, desired_ratio, "downsample")
