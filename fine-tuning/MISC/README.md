# MISC FOLDER

The folder contains three code files:

1. `manual_evaluation.py`
- Used for manually evaluating our finetuned models on random examples.
2. `perplexity_calculator.py`
- This code file helps calcualte perplexity and entropy scores for our synthetic dataset, to ensure that the dataset is ideal for our finetuning use-case.
3. `token_length_counter.py`
- Used for calculating the average, largest and smallest token length of our finetuning data, to ensure that we set the `token_length` limit appropriately.
