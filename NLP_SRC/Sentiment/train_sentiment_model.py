import evaluate
import numpy as np
from preprocessing_sentiment_data import tokenized_sentiment, tokenizer
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

# ── IMPORT YOUR VALIDATED CORE CONFIG ────────────────────────
from configs.core import config

# Extract your validated sentiment model settings object
sent_settings = config.sentiment_model_settings

# ── CELL 7: Model Initialization ─────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    sent_settings.model_checkpoint,
    num_labels=len(sent_settings.label_list),
    id2label=sent_settings.id2label,    # Uses your Pydantic @property dict
    label2id=sent_settings.label2id,    # Uses your Pydantic @property dict
)
print("Model parameters:", model.num_parameters())

# ── CELL 8: Metrics ──────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc          = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1           = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    f1_per_class = f1_metric.compute(predictions=preds, references=labels, average=None)["f1"]

    return {
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1,  4),
        "f1_negative": round(f1_per_class[0], 4),
        "f1_neutral":  round(f1_per_class[1], 4),
        "f1_positive": round(f1_per_class[2], 4),
    }

# ── CELL 9: Training args ────────────────────────────────────
training_args = TrainingArguments(
    output_dir=sent_settings.output_dir,
    num_train_epochs=sent_settings.num_epochs,
    per_device_train_batch_size=sent_settings.batch_size_train,
    per_device_eval_batch_size=sent_settings.batch_size_eval,
    gradient_accumulation_steps=1, # Defaulting to 1 since it's not a specified field
    learning_rate=sent_settings.learning_rate,
    weight_decay=sent_settings.weight_decay,
    warmup_ratio=sent_settings.warmup_ratio,
    lr_scheduler_type="linear",
    fp16=sent_settings.fp16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
)

data_collator = DataCollatorWithPadding(
    tokenizer,  # Assumes tokenizer is initialized earlier in your script
    pad_to_multiple_of=8 if sent_settings.fp16 else None,
)

# ── CELL 10: Train ───────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_sentiment["train"],
    eval_dataset=tokenized_sentiment["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting sentiment training...")
trainer.train()
print("Training complete!")

# ── CELL 11: Evaluate ────────────────────────────────────────
print("\nTest set evaluation:")
test_results = trainer.evaluate(tokenized_sentiment["test"])
for k, v in test_results.items():
    print(f"  {k}: {v}")

# ── CELL 12: Save ────────────────────────────────────────────
FINAL_PATH = f"{sent_settings.output_dir}"
trainer.save_model(FINAL_PATH)
tokenizer.save_pretrained(FINAL_PATH)
print(f"Model saved to: {FINAL_PATH}")