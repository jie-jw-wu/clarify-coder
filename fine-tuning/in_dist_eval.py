from transformers import Trainer, TrainingArguments, EvalPrediction
import numpy as np
from transformers import pipeline

# Initialize the language model pipeline
llm_pipeline = pipeline("text-classification", model="your-llm-model")

# Define the LLM-judge metric function
def llm_judge_metric(preds, labels):
    # Use the LLM to judge the model's output
    judgments = llm_pipeline(preds)
    scores = [judgment['score'] for judgment in judgments]
    avg_score = np.mean(scores)
    return avg_score

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids

    # Assuming preds and labels are lists of strings
    llm_score = llm_judge_metric(preds, labels)
    
    return {
        "llm_judge_score": llm_score,
    }

# Example model and dataset
model = ...  # Your model here
dataset = ...  # Your dataset here

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)




def evaluate_model(model_path, dataset_path, output_dir):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the dataset
    data = load_dataset('json', data_files=dataset_path)
    
    # Tokenize the dataset
    def tokenize_function(samples):
        concatenated_text = samples['problem'] + samples['answer']
        result = tokenizer(
            concatenated_text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_data = data.map(tokenize_function)
    val_dataset = tokenized_data['train'].train_test_split(test_size=0.2, seed=42)['test']

    # Define the Trainer
    training_args = TrainingArguments(
        per_device_eval_batch_size=8,
        output_dir=output_dir,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

    return results
