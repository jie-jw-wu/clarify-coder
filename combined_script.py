# General Imports
import os
import subprocess
import sys

API_KEY = "AIzaSyB4PI14X_caKdryrTukPztiogrBBeSb_9A"
INPUT = "APPS/train"
TYPE = "incomplete"  # Choose either "ambiguous", "inconsistent" or "incomplete"
OUTPUT1 = "INCOMPLETE/train/modified_incomplete_train.jsonl"
MODIFY_PRBLM = "step2.py"
FRMT_CNVR = "convert_JSONL_2_txt.py"
CLARIFY = "step3.py"
INPUT2 = "INCOMPLETE/train"
OUTPUT2 = "INCOMPLETE/train/clarity_incomplete_train.jsonl"
FORMATTER = "step4.py"
PYTHON_VENV = "./venv/Scripts/python.exe"

# Ensure necessary directories exist
os.makedirs(os.path.dirname(OUTPUT1), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT2), exist_ok=True)

print("Starting combined script execution...")

# Generate modified problems from the original dataset
print(f"Running {MODIFY_PRBLM}...")
subprocess.run([PYTHON_VENV, MODIFY_PRBLM, "--api_key", API_KEY, "--dir_path", INPUT,
                "--jsonl_file_path", OUTPUT1, "--type", TYPE], check=True)

# Convert the JSONL file into a format easily processed by the step3 script
print(f"Running {FRMT_CNVR}...")
subprocess.run([PYTHON_VENV, FRMT_CNVR, OUTPUT1, INPUT2], check=True)

# Generate clarifying questions for the modified problems
print(f"Running {CLARIFY}...")
subprocess.run([PYTHON_VENV, CLARIFY, "--api_key", API_KEY, "--dir_path", INPUT2,
                "--jsonl_file_path", OUTPUT2, "--type", TYPE], check=True)

# Generate final finetuning data file
print(f"Running {FORMATTER}...")
subprocess.run([PYTHON_VENV, FORMATTER, OUTPUT2, TYPE], check=True)

print("Combined script execution completed successfully!")