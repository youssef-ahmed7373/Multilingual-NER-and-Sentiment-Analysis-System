from transformers import pipeline

from configs.core import config

# Define the path dynamically matching where your model saved its weights
FINAL_PATH = f"{config.ner_model_settings.output_dir}-final"
import evaluate
from load_ner_data import load_ner_data
from transformers import pipeline

from configs.core import config

test_data=load_ner_data()['test']
FINAL_PATH = f"{config.ner_model_settings.output_dir}-final"

def evaluate_pipeline_on_test_set(test_dataset, model_path: str):
    """
    Evaluates a saved model pipeline using raw text against ground-truth labels.
    """
    print("\nStarting evaluation using Hugging Face Pipeline...")
    
    # 1. Initialize the inference pipeline
    ner_pipe = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="none"  # Crucial: Keep tokens separate to match true labels!
    )
    
    # Load your evaluation metrics metric engine
    seqeval = evaluate.load("seqeval")
    label_list = config.ner_model_settings.label_list
    
    true_labels_all = []
    pred_labels_all = []
    
    # 2. Loop through your dataset records
    for example in test_dataset:
        raw_text = example["text"] if "text" in example else " ".join(example["tokens"])
        
        # Get ground truth integer IDs and map them back to strings (ignoring special tokens like -100)
        true_ids = example["ner_tags"]
        true_strings = [label_list[tid] for tid in true_ids if tid != -100]
        true_labels_all.append(true_strings)
        
        # 3. Get pipeline predictions
        pipe_outputs = ner_pipe(raw_text)
        
        # Extract the string labels predicted by the pipeline
        pred_strings = [ent["entity"] for ent in pipe_outputs]
        
        # Handle length mismatches gracefully if truncation or tokenization differences occur
        if len(pred_strings) != len(true_strings):
            # Pad or truncate predictions to match ground truth array lengths for evaluation alignment
            pred_strings = pred_strings[:len(true_strings)] + ["O"] * max(0, len(true_strings) - len(pred_strings))
            
        pred_labels_all.append(pred_strings)

    # 4. Compute final metric scores
    results = seqeval.compute(predictions=pred_labels_all, references=true_labels_all)
    
    print("\nPipeline Test Results:")
    print(f"  Overall Precision: {results['overall_precision']:.4f}")
    print(f"  Overall Recall:    {results['overall_recall']:.4f}")
    print(f"  Overall F1-Score:  {results['overall_f1']:.4f}")
    print(f"  Overall Accuracy:  {results['overall_accuracy']:.4f}")

    return results


# ── HOW TO RUN IT ────────────────────────────────────
evaluate_pipeline_on_test_set(test_data, FINAL_PATH)