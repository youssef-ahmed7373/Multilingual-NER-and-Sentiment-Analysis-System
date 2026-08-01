import json
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import sys
from pathlib import Path

from load_sentiment_data import load_sentiment_data

# ── IMPORT YOUR VALIDATED CORE CONFIG ────────────────────────
from configs.core import config

languages=config.sentiment_model_settings.languages
train_samples=config.sentiment_model_settings.train_samples
val_samples=config.sentiment_model_settings.val_samples
test_samples=config.sentiment_model_settings.test_samples

raw_sentiment=load_sentiment_data(languages,
                                  train_samples,
                                  val_samples,
                                  test_samples)['test']

def main():
    # Extract your validated sentiment model settings object
    sent_settings = config.sentiment_model_settings
    
    # Resolve the path to your final trained weights
    base_dir = Path(sent_settings.output_dir).parent
    FINAL_PATH = base_dir / "final"
    METRICS_OUTPUT_PATH = FINAL_PATH / "pipeline_test_metrics.json"
    
    if not FINAL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at: {FINAL_PATH}")

    # Determine execution device (-1 for CPU, 0 or greater for CUDA GPU device ID)
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Initializing inference pipeline on device: {'GPU (0)' if device_id >= 0 else 'CPU'}")

    # ── INITIALIZE HUGGING FACE PIPELINE ─────────────────────────
    # The pipeline automatically pairs your saved weights, tokenizer, config mappings, and device management
    classifier = pipeline(
        task="sentiment-analysis",
        model=str(FINAL_PATH),
        tokenizer=str(FINAL_PATH),
        device=device_id,
        batch_size=sent_settings.batch_size_eval
    )

    # ── PREPARE TEST SAMPLES ─────────────────────────────────────
    # NOTE: Replace 'tokenized_sentiment' references with your actual dataset pipeline.
    # Because pipeline accepts raw strings, we pull the original text strings and ground-truth labels.
    # E.g., if using a Hugging Face Dataset object:
    # test_dataset =raw_sentiment['test']
    raw_texts = raw_sentiment["text"]
    true_labels = raw_sentiment["label"]

    print(f"Running pipeline inference over {len(raw_texts)} test cases...")

    # ── RUN INFERENCE ────────────────────────────────────────────
    # Passing a list of strings to the pipeline leverages batched GPU evaluation
    predictions = classifier(raw_texts)

    # ── PROCESS PREDICTIONS ──────────────────────────────────────
    # Map text string outputs (e.g., "positive") back to integer IDs using your Pydantic property map
    label2id = sent_settings.label2id
    predicted_ids = [label2id[pred["label"]] for pred in predictions]

    # ── CALCULATE PERFORMANCE METRICS ────────────────────────────
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    acc          = accuracy_metric.compute(predictions=predicted_ids, references=true_labels)["accuracy"]
    f1           = f1_metric.compute(predictions=predicted_ids, references=true_labels, average="weighted")["f1"]
    f1_per_class = f1_metric.compute(predictions=predicted_ids, references=true_labels, average=None)["f1"]

    cleaned_results = {
        "test_accuracy":    round(acc, 4),
        "test_f1_weighted": round(f1,  4),
        "test_f1_negative": round(f1_per_class[0], 4),
        "test_f1_neutral":  round(f1_per_class[1], 4),
        "test_f1_positive": round(f1_per_class[2], 4),
    }

    # ── DISPLAY AND CACHE RESULTS ────────────────────────────────
    print("\n================ PIPELINE TEST PERFORMANCE ================")
    for key, val in cleaned_results.items():
        print(f" {key.upper()}: {val}")
    print("===========================================================\n")

    with open(METRICS_OUTPUT_PATH, "w") as json_file:
        json.dump(cleaned_results, json_file, indent=4)
        
    print(f"Evaluation metrics cached successfully to: {METRICS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()