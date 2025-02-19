#!/bin/bash

API_KEY="<ENTER_GEMINI_API_KEY>"
INPUT="APPS/train"
TYPE="ambiguous"  # Choose either "ambiguous", "inconsistent" or "incomplete"
OUTPUT1="AMBIGUOUS/train/modified_ambiguous_train.jsonl"
MODIFY_PRBLM="step2.py"
FRMT_CNVR="convert_JSONL_2_txt.py"
CLARIFY="step3.py"
INPUT2="AMBIGUOUS/train"
OUTPUT2="AMBIGUOUS/train/clarity_ambiguous_train.jsonl"
FORMATTER="step4.py"

# Ensure necessary directories exist
mkdir -p "$(dirname "$OUTPUT1")"
mkdir -p "$(dirname "$OUTPUT2")"

# Generate modified problems from the original dataset
python3 $MODIFY_PRBLM --api_key $API_KEY --dir_path $INPUT --jsonl_file_path $OUTPUT1 --type $TYPE

# Convert the JSONL file into a format easily processed by the step3 script
python3 $FRMT_CNVR $OUTPUT1 $INPUT2

# Generate clarifying questions for the modified problems
python3 $CLARIFY --api_key $API_KEY --dir_path $INPUT2 --jsonl_file_path $OUTPUT2 --type $TYPE

# Generate final finetuning data file
python3 $FORMATTER $OUTPUT2 $TYPE