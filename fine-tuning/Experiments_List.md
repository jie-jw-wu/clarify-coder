# Finetuning Experiment Details

- **Finetuning Data**: 
    - The training data consists of the following number of samples:
        - Original: 8765 (from the APPS train and test sets)
        - Ambiguous: 9968 (Gemini generated clarifying questions)
        - Inconsistent: 9990 (Gemini generated clarifying questions)
        - Incomplete: 9938 (Gemini generated clarifying questions)
    - The prompts used to generate these clarifying questions for the three kinds of modified problems are as follows:

    - **Ambiguous**
    ```
    You are given a coding problem description and an ambiguous version of the same problem. Your task is to assess the ambiguous problem description, identify specific points of ambiguity, and ask necessary clarifying questions to resolve the ambiguity and reach the clarity present in the original problem. Please note that a problem statement is ambiguous if it includes multiple valid interpretations or has unspecified details.

    Original Coding Problem Description: <>

    Ambiguous Coding Problem Description: <>

    When generating these questions, do not mention the original problem description in any way. Frame the clarifying questions without acknowledging the existence of the original version. 

    Please only output the clarifying questions, ensuring that each question targets a specific point of ambiguity to achieve clarity similar to the original problem.
    
    ``` 
    - **Inconsistent**
    ```
    You are given a coding problem description and an inconsistent version of the same problem. Your task is to assess the inconsistent problem description, identify specific points of inconsistency, and ask necessary clarifying questions to resolve the inconsistency and reach the clarity present in the original problem. Please note that a problem description becomes inconsistent if some statements in the description show conflict.

    Original Coding Problem Description: <>

    Inconsistent Coding Problem Description: <> 

    When generating these questions, do not mention the original problem description in any way. Frame the clarifying questions without acknowledging the existence of the original version. 

    Please only output the clarifying questions, ensuring that each question targets a specific point of inconsistency to achieve clarity similar to the original problem.
    ``` 
    - **Incomplete**
    ```
    You are given a coding problem description and an incomplete version of the same problem. Your task is to assess the incomplete problem description, identify specific points of incompleteness, and ask necessary clarifying questions to resolve the incompleteness and reach the clarity present in the original problem. Please nore that absence of some of the key concepts and conditions that are crucial for solving the problem makes it incomplete.

    Original Coding Problem Description: <>

    Incomplete Coding Problem Description: <>

    When generating these questions, do not mention the original problem description in any way. Frame the clarifying questions without acknowledging the existence of the original version.

    Please only output the clarifying questions, ensuring that each question targets a specific point of incompleteness to achieve clarity similar to the original problem.
    ``` 

- **Code**
    - We use the `clarify_aware_fine_tuning_v2.py` file, varying the `--tokenize_version` argument to test different approaches. 

- **Model**
    - Our baseline is the `deepseek-ai/deepseek-coder-6.7b-instruct` model. Due to compute restrictions, we use the 7B model with PEFT enabled.

## Tokenizer V1
- We set the `--tokenize_version` argument to `1`, passing the concatenation of the problem and answer to the model for fine-tuning.
```
concatenated_text = samples['problem'] + samples['answer']
```

## Tokenizer V2
- We set the `--tokenize_version` argument to `2`, focusing only on the "answer" part of the dataset while finetuning.
- Basically, we tokenize both the "problem" and its corresponding "answer" while preparing it for the model, but in the training setup, only the "answer" part is important for computing the loss. The "problem" is provided for context but doesn't contribute to the loss.

## Tokenizer V3
- We set the `--tokenize_version` argument to `3`, passing the concatenation of the problem, answer and type to the model for fine-tuning.
```
concatenated_text = samples['problem'] + samples['answer'] + samples['type']
```

## Tokenizer V4 (R-Tuning Inspired Instruction-Tuning Approach)
- We set the `--tokenize_version` argument to `4`, instruction-tuning the model to effectively classify if the problem requires code or questions, and then generate accordingly.
- The function uses two different prompts based on the type of sample:
    - QPROMPT: A constant instruction to the model to act as an expert software developer. This sets up a scenario where the model is expected to either generate code or ask clarifying questions.
    - APROMPT: A conditional response based on the sample's type (Original or otherwise). If the sample is labeled as "Original," the model is instructed that the problem is clear, and it should directly respond with Python code. Otherwise, the model is instructed to ask clarifying questions before proceeding with code generation.
- The structured prompts ensure that the model can differentiate between when it should generate code directly and when it should seek further clarification. This controlled format improves the quality of responses during inference, as the model is guided through a clear decision-making process.