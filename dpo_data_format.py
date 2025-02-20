import json
import os


input_file = "output_splits/split_20_80_downsample.jsonl"
output_folder = "dpo_data"
output_file = os.path.join(output_folder, "dpo_finetune_data.jsonl")

os.makedirs(output_folder, exist_ok=True)

formatted_data = []

def generate_worse_answer(correct_answer):
    return correct_answer.replace("return", "# return (incorrectly commented)") if "return" in correct_answer else "I'm not sure how to solve this."

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        
        if "problem" in data and "answer" in data:
            prompt = f"Problem: {data['problem']}\n\nSolution:"
            chosen = data["answer"]
            rejected = generate_worse_answer(chosen)  

            formatted_entry = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            formatted_data.append(formatted_entry)


with open(output_file, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")

print(f"Formatted data saved to {output_file} with {len(formatted_data)} entries.")
