import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the local paths for the model and tokenizer
MODEL_PATH = "finetuned_models/ANAD_deepseek-7B-exp6-bin"
TOKENIZER_PATH = "deepseek-ai/deepseek-coder-6.7b-instruct" 

# Load the tokenizer from the local path
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Load the fine-tuned model from the local path
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# Ensure the model is in evaluation mode
model.eval()

# Function to generate outputs manually
def generate_response(input_text, max_length=2048, temperature=0.7):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Generate a response using the model
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
input_text = "Chingel is practicing for a rowing competition to be held on this saturday. He is trying his best to win this tournament for which he needs to figure out how much time it takes to cover a certain distance. \n\n**Input**\n\nYou will be provided with the total distance of the journey and speed of the boat.\n\n**Output**\n\nThe output returned should be the time taken to cover the distance. If the result has decimal places, round them to 2 fixed positions.\n\n`Show some love ;) Rank and Upvote!"

response = generate_response(input_text)

# print(f"Input: {input_text}\n")
print(f"Generated Response: {response}\n")
