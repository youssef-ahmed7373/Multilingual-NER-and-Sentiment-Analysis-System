# Import the warm layout validation instance from your core module
from pathlib import Path

import evaluate
import numpy as np
from preprocessing_ner_data import tokenized_ner, tokenizer
from transformers import (AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from configs.core import config

# ── MODEL INITIALIZATION ─────────────────────────────────────
# Accessing all settings safely from the validated Pydantic model
model = AutoModelForTokenClassification.from_pretrained(
    config.ner_model_settings.model_checkpoint,
    num_labels=len(config.ner_model_settings.label_list),
    id2label=config.ner_model_settings.id2label,      # Evaluates from schema @property
    label2id=config.ner_model_settings.label2id,      # Evaluates from schema @property
)
print("Model parameters:", model.num_parameters())

# ── CELL 7: Metrics ──────────────────────────────────────────
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Use the label list directly from settings
    label_list = config.ner_model_settings.label_list

    true_labels = [
        [label_list[l] for l in row if l != -100]
        for row in labels
    ]
    true_preds = [
        [label_list[p] for p, l in zip(pr, lr) if l != -100]
        for pr, lr in zip(predictions, labels)
    ]
    r = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": round(r["overall_precision"], 4),
        "recall":    round(r["overall_recall"],    4),
        "f1":        round(r["overall_f1"],        4),
        "accuracy":  round(r["overall_accuracy"],  4),
    }

# ── CELL 8: Training args ────────────────────────────────────
# Note: Since GRAD_ACCUM_STEPS wasn't explicitly defined in your YAML, 
# it defaults to 1 here, but you can add it to your yaml settings if needed!
training_args = TrainingArguments(
    output_dir=config.ner_model_settings.output_dir,
    num_train_epochs=config.ner_model_settings.num_epochs,
    per_device_train_batch_size=config.ner_model_settings.batch_size_train,
    per_device_eval_batch_size=config.ner_model_settings.batch_size_eval,
    gradient_accumulation_steps=1, 
    learning_rate=config.ner_model_settings.learning_rate,
    weight_decay=config.ner_model_settings.weight_decay,
    warmup_ratio=config.ner_model_settings.warmup_ratio,
    fp16=config.ner_model_settings.fp16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
)

data_collator = DataCollatorForTokenClassification(
    tokenizer,  # Assumes tokenizer was loaded using config.ner_model_settings.model_checkpoint
    pad_to_multiple_of=8 if config.ner_model_settings.fp16 else None,
)

# ── CELL 9: Train ────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ner["train"],
    eval_dataset=tokenized_ner["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting NER training...")
trainer.train()
print("Training complete!")

FINAL_PATH = Path(config.ner_model_settings.output_dir)
# FINAL_PATH = base_output_dir.parent / "final"
FINAL_PATH.mkdir(parents=True, exist_ok=True)

trainer.save_model(FINAL_PATH)
tokenizer.save_pretrained(FINAL_PATH)
print(f"Model saved to: {FINAL_PATH}")