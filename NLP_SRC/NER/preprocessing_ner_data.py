import os
import sys
from pathlib import Path

from load_ner_data import load_ner_data
from transformers import AutoTokenizer

project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(str(project_root))

from configs.core import config

PROCESSED_DIR=Path(config.ner_model_settings.processed_data_dir)


raw_ner = load_ner_data()
print("\nDataset loaded:", raw_ner)
print("Example:", raw_ner["train"][0])


tokenizer = AutoTokenizer.from_pretrained(config.ner_model_settings.model_checkpoint)
print("Tokenizer loaded:", config.ner_model_settings.model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=config.ner_model_settings.max_length,
        is_split_into_words=True,
        padding="max_length",
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids  = tokenized_inputs.word_ids(batch_index=i)
        aligned, prev_word = [], None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)
            prev_word = word_id
        all_labels.append(aligned)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

print("Tokenizing...")
tokenized_ner = raw_ner.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_ner["train"].column_names,
)

print(f"💾 Caching processed arrow layers to: {PROCESSED_DIR}")
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
tokenized_ner.save_to_disk(str(PROCESSED_DIR))
print(f'NER Processed Data Saved Local.in {PROCESSED_DIR}')
print("Done. Columns:", tokenized_ner["train"].column_names)

