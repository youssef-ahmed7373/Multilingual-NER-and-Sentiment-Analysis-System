
import sys
from pathlib import Path

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Inject the root into Python's search path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configs.core import config

RAW_DIR=Path(config.ner_model_settings.raw_data_dir)

# ── CELL 4: Load data ────────────────────────────────────────
def load_ner_data(languages=config.ner_model_settings.languages, train_samples=config.ner_model_settings.train_samples, val_samples=config.ner_model_settings.val_samples, test_samples=config.ner_model_settings.test_samples):
    train_splits, val_splits, test_splits = [], [], []

    for lang in languages:
        print(f"Loading WikiANN — language: {lang}")
        ds = load_dataset("unimelb-nlp/wikiann", lang)

        n_train = min(train_samples, len(ds["train"]))      if train_samples else len(ds["train"])
        n_val   = min(val_samples,   len(ds["validation"])) if val_samples   else len(ds["validation"])
        n_test  = min(test_samples,  len(ds["test"]))       if test_samples  else len(ds["test"])

        train_splits.append(ds["train"].shuffle(42).select(range(n_train)))
        val_splits.append(ds["validation"].shuffle(42).select(range(n_val)))
        test_splits.append(ds["test"].shuffle(42).select(range(n_test)))

        print(f"  train: {n_train} | val: {n_val} | test: {n_test}")

        merged_data=DatasetDict({
        "train":      concatenate_datasets(train_splits),
        "validation": concatenate_datasets(val_splits),
        "test":       concatenate_datasets(test_splits),
        })
        RAW_DIR.mkdir(parents=True,exist_ok=True)
        
        print(f"💾 Caching immutable raw data assets to: {RAW_DIR}")
        merged_data.save_to_disk(str(RAW_DIR))
        print(f'NER Raw Data Saved Local in {RAW_DIR}')


    return merged_data