# clarify-aware-coder

## Data Generation Pipeline
- This repository contains scripts and resources for generating modified coding problem descriptions (ambiguous, inconsistent, or incomplete) from the APPS dataset and subsequently generating clarifying questions for these modified problems. The process involves several steps, each executed by specific scripts. The end-to-end pipeline is designed to work with the Gemini language model and includes mechanisms to handle potential errors and interruptions.
### Datasets
#### APPS Dataset
- The APPS dataset consists of problems collected from different open-access coding websites such as Codeforces, Kattis, and more. The problems range in difficulty from introductory to collegiate competition level and measure coding ability as well as problem-solving. 
- The Automated Programming Progress Standard, abbreviated APPS, consists of 10,000 coding problems in total, with 131,836 test cases for checking solutions and 232,444 ground-truth solutions written by humans. Problems can be complicated, as the average length of a problem is 293.2 words. The data are split evenly into training and test sets, with 5,000 problems each. In the test set, every problem has multiple test cases, and the average number of test cases is 21.2. Each test case is specifically designed for the corresponding problem, enabling us to rigorously evaluate program functionality.
- Download the APPS dataset from the following link: https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
- The folder structure:
```
APPS
├── train
│   ├── 0000
│   │   ├── input_output.json
│   │   ├── metadata.json
│   │   ├── question.txt
│   │   └── solutions.json
│   └── ...
└── test
    ├── 0000
    │   ├── input_output.json
    │   ├── metadata.json
    │   ├── question.txt
    │   └── solutions.json
    └── ...

```

### LLMs
#### Gemini
- We use Google Gemini to generate our dataset. Why? Great performance and free access to API for research purposes.
- As such, we use imports such as google.generativeai to make calls to the model, and google.api_core.exceptions to handle errors.
- **The API_KEY can be generated from the following link: https://aistudio.google.com/app/apikey.** Please note that you will need a Google Account to get access to an API key.
- Current Configuration Settings:
    - Temperature: 0.1
    - Top_p: 1
    - Top_k: 1
    - Maximum output tokens: 2048
    - Safety settings set to “BLOCK_NONE”.
### Running the Data Generation Pipeline in one-go
- Clone the repository and navigate to the `clarity-aware-code` directory.
- Ensure that your dataset folder is in the same directory as the python files as well as the batch script.
- To run the entire process sequentially, you can use the following batch script (also included in the repository titled `combined_script.sh`):
```
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

# Ensure necessary directories exist
mkdir -p "$(dirname "$OUTPUT1")"
mkdir -p "$(dirname "$OUTPUT2")"

# Generate modified problems from the original dataset
python3 $MODIFY_PRBLM --api_key $API_KEY --dir_path $INPUT --jsonl_file_path $OUTPUT1 --type $TYPE

# Convert the JSONL file into a format easily processed by the step3 script
python3 $FRMT_CNVR $OUTPUT1 $INPUT2

# Generate clarifying questions for the modified problems
python3 $CLARIFY --api_key $API_KEY --dir_path $INPUT2 --jsonl_file_path $OUTPUT2 --type $TYPE

```
- Save this script to a file, e.g., combined_script.sh.
- Make the script executable:
```
chmod +x combined_script.sh
```
- Run the script:
```
./combined_script.sh
```
- Ensure Python is installed on your local system and that the required Python packages are available globally. If you need to install specific packages, you can do so using pip:
```
pip install <package_name>
```
- **Recommendation**: Due to the possibility of errors and interruptions, it is advisable to perform the process step-by-step, especially for generating large datasets. The checkpoint mechanism helps in continuing the process without losing progress while the `update_empty_responses` function  reads the generated JSONL output and sends queries to the model for those where the output is empty.

### Detailed step-by-step Data Generation Guide

#### Step 2

- Generates modified problem from the original problem statement.
```
python3 step2.py --api_key <API_KEY> --dir_path APPS/train --jsonl_file_path AMBIGUOUS/train/modified_ambiguous_train.jsonl --type ambiguous
```
- Input: question.txt files from the APPS dataset.
- Output: A JSONL file containing prompt + original problem as input and the modified problems as output based on the chosen type (ambiguous, inconsistent, or incomplete).

#### Convert JSONL to Text (Step 2.5)

- To prepare the output from the previous step for the next stage, convert the JSONL file to text format.
```
python3 convert_JSONL_2_txt.py AMBIGUOUS/train/modified_ambiguous_train.jsonl AMBIGUOUS/train
```
- Output: A set of folders, each containing a modified_question.txt file. This is what the output should look like:
```
Modified_Problems
├── 0
│   └── modified_question.txt
├── 1
│   └── modified_question.txt
└── ...

```

#### Step 3

- Generates clarifying questions for the modified problems.
```
python3 step3.py --api_key <API_KEY> --dir_path AMBIGUOUS/train --jsonl_file_path AMBIGUOUS/train/clarity_ambiguous_train.jsonl --type ambiguous
```
- Output: A JSONL file with input-output pairs, where the input is the prompt + modified problem description and the output is the clarifying questions.

### Error handling

- The scripts include mechanisms to handle common errors:
    - Empty Output: The function `update_empty_responses` reads the generated JSONL output and sends queries to the model for those where the output is empty, handling errors like 403: Couldn’t generate even after delay.
    - <ADD HOW TO USE>
    - Internal Errors: If an error occurs and generation stops (e.g., 500: An Internal Error has Occurred), a checkpoint.txt file is used to track progress. The script checks this file upon starting and resumes from the last successful index.
    - <ADD HOW TO USE>