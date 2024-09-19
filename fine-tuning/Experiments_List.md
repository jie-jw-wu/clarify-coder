# Finetuning Experiment Details

## Experiment 1 (the baseline)

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

    When generating these questions, do not reference or mention the original problem description in any way. Frame the clarifying questions as if you have only seen the modified problem description, without acknowledging the existence of the original version. 

    Please only output the clarifying questions, ensuring that each question targets a specific point of ambiguity to achieve clarity similar to the original problem.
    
    ``` 
    - **Inconsistent**
    ```
    You are given a coding problem description and an inconsistent version of the same problem. Your task is to assess the inconsistent problem description, identify specific points of inconsistency, and ask necessary clarifying questions to resolve the inconsistency and reach the clarity present in the original problem. Please note that a problem description becomes inconsistent if some statements in the description show conflict.

    Original Coding Problem Description: <>

    Inconsistent Coding Problem Description: <> 

    When generating these questions, do not reference or mention the original problem description in any way. Frame the clarifying questions as if you have only seen the modified problem description, without acknowledging the existence of the original version.

    Please only output the clarifying questions, ensuring that each question targets a specific point of inconsistency to achieve clarity similar to the original problem.
    ``` 
    - **Incomplete**
    ```
    You are given a coding problem description and an incomplete version of the same problem. Your task is to assess the incomplete problem description, identify specific points of incompleteness, and ask necessary clarifying questions to resolve the incompleteness and reach the clarity present in the original problem. Please nore that absence of some of the key concepts and conditions that are crucial for solving the problem makes it incomplete.

    Original Coding Problem Description: <>

    Incomplete Coding Problem Description: <>

    When generating these questions, do not reference or mention the original problem description in any way. Frame the clarifying questions as if you have only seen the modified problem description, without acknowledging the existence of the original version. 

    Please only output the clarifying questions, ensuring that each question targets a specific point of incompleteness to achieve clarity similar to the original problem.
    ``` 

- **Code**
    - We use the `clarify_aware_fine_tuning_v2.py` file.

- **Model**
    - We use the `deepseek-ai/deepseek-coder-6.7b-instruct` model. Due to compute restrictions, we can only use the 7B model.

## Experiment 2 (Integration of "type")
- **Finetuning Data**: We augment the original dataset by adding the following prefixes to the answer fields:
    - Original: The problem statement is straightforward and requires no additional information. We can proceed with the implementation without further questions.
    - Ambiguous: The problem statement is not fully clear, and additional details are needed to proceed effectively. This is an ambiguous problem statement with multiple valid interpretations and unspecified details, requiring clarification.",
    - Inconsistent: The problem statement is not fully clear, and additional details are needed to proceed effectively. This is an inconsistent problem statement and contains conflicting information that must be clarified.
    - Incomplete: The problem statement is not fully clear, and additional details are needed to proceed effectively. This is an incomplete problem statement and it lacks some crucial details that need to be filled in to create a complete solution.

- **Code**
    - We use the `clarify_aware_fine_tuning_v3.py` file. The only change is in the `tokenize` function, where we concatenate `type` field into the final output.
    ```
        concatenated_text = samples['problem'] + samples['answer'] + samples['type']
    ```
- **Model**
    - Same as experiment 1, we use the `deepseek-ai/deepseek-coder-6.7b-instruct` model.

## Experiment 3 (Focusing on the "answer")
- **Finetuning Data**: Same as Experiment 2.
- **Code**
    - We use the `clarify_aware_fine_tuning_v4.py` file. We modify the `tokenize` function to focus on the `answer` field. We use formatting tokens to help the model learn the association between different problem types and expected outputs. If the problem is unclear (Ambiguous, Incomplete, Inconsistent), then the model is expected to generate clarifying questions, otherwise it should generate code.
    ```
        def tokenize(samples):
        type = samples['type']
        if type == "Original":
            answer = f"[CODE] {samples['answer']}"
        else:
            answer = f"[QUESTION] {samples['answer']}"

        concatenated_text = f"Problem Type: {type}\nProblem: {samples['problem']}\nAnswer:"
        
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )

        result["labels"] = tokenizer(
            answer, 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )["input_ids"]

        return result

    ```
- **Model**
    - Same as experiment 1, we use the `deepseek-ai/deepseek-coder-6.7b-instruct` model.

## Experiment 4 (Classification Layer)

## Experiment 5 (Chat Model)

## Experiment 6 (Twice as many parameters (almost) : 13B)