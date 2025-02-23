import json
import os
import google.generativeai as genai

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-pro")


input_file = "output_splits/split_20_80_downsample.jsonl"
output_folder = "dpo_data"
output_file = os.path.join(output_folder, "dpo_finetune_data.jsonl")

os.makedirs(output_folder, exist_ok=True)

formatted_data = []

def generate_worse_answer(correct_answer, type):
    if type != "Original":
        prompt = ("Given the following correct code, generate an incorrect version that contains logical or syntactical errors: \n\n" + correct_answer)
        response = model.generate_content(prompt)
        
    else:
        prompt = ("Analyze the following code and generate clarifying questions that help identify ambiguities, edge cases, or missing details in its functionality, logic, or intended use:\n\n" + correct_answer)        
        response = model.generate_content(prompt)
    
    return response.text if response.text else "Error generating incorrect code"

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        data = json.loads(line)
        
        if "problem" in data and "answer" in data:
            prompt = f"Problem: {data['problem']}\n\nSolution:"
            chosen = data["answer"]
            rejected = generate_worse_answer(chosen, data['type'])  
            data_type = data['type']

            formatted_entry = {
                "type" : data_type,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            formatted_data.append(formatted_entry)


with open(output_file, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")

print(f"Formatted data saved to {output_file} with {len(formatted_data)} entries.")
