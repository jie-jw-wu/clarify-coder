#!/bin/bash
#SBATCH --time=<choose in hh:mm:ss format>
#SBATCH --account=<your_account>
#SBATCH --mem-per-cpu=<doesnt require much, 1G should suffice>
#SBATCH --ntasks=1

source GeminiTest1/bin/activate # to activate environment

API_KEY="<ENTER_GEMINI_API_KEY>"
INPUT="APPS/train"
TYPE="ambiguous" # choose either "ambiguous", "inconsistent" or "incomplete" prompt to send to LLM
OUTPUT1="AMBIGUOUS/train/modified_ambiguous_train.jsonl"
MODIFY_PRBLM="step2.py"
FRMT_CNVR="convert_JSONL_2_txt.py"
CLARIFY="step3.py"
INPUT2="AMBIGUOUS/train"
OUTPUT2="AMBIGUOUS/train/clarity_ambiguous_train.jsonl"

# this step generates modified problems from the original dataset.
python3 $MODIFY_PRBLM --api_key $API_KEY --dir_path $INPUT --jsonl_file_path $OUTPUT1 --type $TYPE

# this step converts the jsonl file into a format easily processed by step3 script
python3 $FRMT_CNVR $OUTPUT1 $INPUT2

# this step generates clarifying questions for the modified problems.
python3 $CLARIFY --api_key $API_KEY --dir_path $INPUT2 --jsonl_file_path $OUTPUT2 --type $TYPE