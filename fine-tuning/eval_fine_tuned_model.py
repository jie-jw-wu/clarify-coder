import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Path to the fine-tuned model bin file
MODEL_PATH = "/home/jie/clarify-aware-coder/fine-tuning/ANAD_deepseek-7B-exp4-bin-20241001T173940Z-001/ANAD_deepseek-7B-exp4-bin"  # Replace with the correct path


# Load the tokenizer and model from Hugging Face's pre-trained base
# We assume that your fine-tuned model uses the same tokenizer as the base model (deepseek-ai/deepseek-coder-6.7b-instruct)
tokenizer = AutoTokenizer.from_pretrained("/project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct")


# Load the fine-tuned model (this will load the checkpoint from your bin file)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")


# Ensure the model is in evaluation mode (not training)
model.eval()


# Function to generate outputs manually
def generate_response(input_text, max_length=512, temperature=0.7):
   # Tokenize input
   inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
  
   # Generate response using the model
   with torch.no_grad():
       outputs = model.generate(
           inputs["input_ids"],
           max_length=max_length,
           temperature=temperature,  # Controls randomness in generation
           num_return_sequences=1,   # Generate only one response
           no_repeat_ngram_size=2,   # Avoid repeating n-grams
           pad_token_id=tokenizer.eos_token_id
       )
  
   # Decode generated output to text
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   return response


# Manual check: Input example (you can modify this)
input_text = "Digory and Polly are two kids living next door to each other. The attics of the two houses are connected to each other through a passage. Digory's Uncle Andrew has been secretly doing strange things in the attic of his house, and he always ensures that the room is locked. Being curious, Digory suspects that there is another route into the attic through Polly's house, and being curious as kids always are, they wish to find out what it is that Uncle Andrew is secretly up to.\n\nSo they start from Polly's house, and walk along the passageway to Digory's. Unfortunately, along the way, they suddenly find that some of the floorboards are missing, and that taking a step forward would have them plummet to their deaths below.\n\nDejected, but determined, they return to Polly's house, and decide to practice long-jumping in the yard before they re-attempt the crossing of the passage. It takes them exactly one day to master long-jumping a certain length. Also, once they have mastered jumping a particular length L, they are able to jump any amount less than equal to L as well.\n\nThe next day they return to their mission, but somehow find that there is another place further up the passage, that requires them to jump even more than they had practiced for. So they go back and repeat the process.\n\nNote the following:\n\n-  At each point, they are able to sense only how much they need to jump at that point, and have no idea of the further reaches of the passage till they reach there. That is, they are able to only see how far ahead is the next floorboard. \n-  The amount they choose to practice for their jump is exactly the amount they need to get across that particular part of the passage. That is, if they can currently jump upto a length L0, and they require to jump a length L1(> L0) at that point, they will practice jumping length L1 that day. \n-  They start by being able to \"jump\" a length of 1. \n\nFind how many days it will take them to cross the passageway. In the input, the passageway is described as a string P of '#'s and '.'s. A '#' represents a floorboard, while a '.' represents the absence of a floorboard. The string, when read from left to right, describes the passage from Polly's house to Digory's, and not vice-versa.\n\n-----Input-----\n\nThe first line consists of a single integer T, the number of testcases.\nEach of the next T lines consist of the string P for that case.\n\n-----Output-----\n\nFor each case, output the number of days it takes them to cross the passage.\n\n-----Constraints-----\n-  1  \u2264 T  \u2264 1,000,000  (106)\n-  1  \u2264 |P|  \u2264 1,000,000 (106)\n-  The total length of P will be \u2264 5,000,000 (5 * 106)across all test-cases of a test-file \n-  P will consist of only the characters # and . \n-  The first and the last characters of P will be #. \n\n-----Example-----\nInput:\n4\n####\n##.#..#\n##..#.#\n##.#....#\n\nOutput:\n0\n2\n1\n2\n\n-----Explanation-----\n\nFor the first example, they do not need to learn any jump size. They are able to cross the entire passage by \"jumping\" lengths 1-1-1.\n\nFor the second example case, they get stuck at the first '.', and take one day learning to jump length 2. When they come back the next day, they get stuck at '..' and take one day to learn to jump length 3.\n\nFor the third example case, they get stuck first at '..', and they take one day to learn to jump length 3. On the second day, they are able to jump both length 3 as well as length 2 required to cross the passage.\n\nFor the last test case they need to stop and learn jumping two times. At first they need to jump a length 2 and then a length 5.\n\n-----Appendix-----\n\nIrrelevant to the problem description, if you're curious about what Uncle Andrew was up to, he was experimenting on Magic Rings that could facilitate travel between worlds. One such world, as some of you might have heard of, was Narnia.\n\n-----Ambiguous Part-----\n\nThe problem description does not specify whether the kids can see the entire passageway at once or only the part they are currently at. This ambiguity can lead to different interpretations of the problem and different solutions."
response = generate_response(input_text)


#print(f"Input: {input_text}\n")
print(f"Generated Response: {response}\n")