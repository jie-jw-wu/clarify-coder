# model_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# functions in this file are copied from https://github.com/jie-jw-wu/human-eval-comm/, in case any one of them is used in the future (not used yet), please refer to the original source
def load_model(args):
    HF_HOME = args.hf_dir
    offload_folder = "offload_folder"
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    print("Loading model...")
    if args.do_save_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path      
        )
    elif args.use_int8:
        print("**********************************")
        print("**** Using 8-bit quantization ****")
        print("**********************************")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=HF_HOME,
            offload_folder=offload_folder,     
        )
    elif args.use_fp16:
        print("**********************************")
        print("****** Using fp16 precision ******")
        print("**********************************")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=HF_HOME,
            offload_folder=offload_folder,     
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            cache_dir=HF_HOME,
            offload_folder=offload_folder,            
        )

    if 'finetuned' in args.model:
        model = PeftModel.from_pretrained(model, args.finetuned_model_path)

    print('model device: ', model.device)
    return model

def load_tokenizer(args):
    HF_HOME = args.hf_dir
    offload_folder = "offload_folder"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.seq_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=HF_HOME,
        offload_folder=offload_folder,
    )
    return tokenizer

def test_model(tokenizer, model, user_input, max_length):
    timea = time.time()
    input_ids = tokenizer(user_input, return_tensors="pt")["input_ids"].to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=max_length)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    print('!!!!!!!!!!')
    print(filling)
    print('!!!!!!!!!!')
    print("timea = time.time()", -timea + time.time())
