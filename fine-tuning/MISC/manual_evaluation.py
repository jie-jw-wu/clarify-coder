import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "<FINETUNED MODEL PATH>"
TOKENIZER_PATH = "deepseek-ai/deepseek-coder-6.7b-instruct" # adjust as necessary

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

model.eval()

def generate_response(input_text, max_length=512, temperature=0.2): # adjust settings as necessary

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # generate a response using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,  # Controls randomness in generation
            num_return_sequences=1,   # Generate only one response
            no_repeat_ngram_size=2,   # Avoid repeating n-grams
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated output to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input text from the generated output
    response = generated_text[len(input_text):].strip()
    return response

# Test the model with an example input
input_text = "Write code that either prints YES if input number is even and NO if it is even."

response = generate_response(input_text)

# print(f"Input: {input_text}\n")
print(f"Generated Response: {response}\n")
