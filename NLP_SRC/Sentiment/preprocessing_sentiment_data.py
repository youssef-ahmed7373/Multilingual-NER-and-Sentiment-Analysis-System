from pathlib import Path

from load_sentiment_data import load_sentiment_data
from transformers import AutoTokenizer

from configs.core import config

languages=config.sentiment_model_settings.languages
train_samples=config.sentiment_model_settings.train_samples
val_samples=config.sentiment_model_settings.val_samples
test_samples=config.sentiment_model_settings.test_samples

raw_sentiment=load_sentiment_data(languages,
                                  train_samples,
                                  val_samples,
                                  test_samples)
PROCESSED_DIR=Path(config.sentiment_model_settings.processed_data_dir)

tokenizer = AutoTokenizer.from_pretrained(config.sentiment_model_settings.model_checkpoint)

def tokenize_sentiment(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.sentiment_model_settings.max_length,
        padding="max_length",
    )

print("\nTokenizing...")
tokenized_sentiment = raw_sentiment.map(
    tokenize_sentiment,
    batched=True,
    remove_columns=["text"],
)

print(f"💾 Caching processed arrow layers to: {PROCESSED_DIR}")
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
tokenized_sentiment.save_to_disk(str(PROCESSED_DIR))
print(f'Sentiment Processed Data Saved Local in {PROCESSED_DIR}')
print("Done. Columns:", tokenized_sentiment["train"].column_names)