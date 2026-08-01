import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Inject the root into Python's search path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import evaluate

from configs.core import config

RAW_DIR=Path(config.sentiment_model_settings.raw_data_dir)

def load_sentiment_data(languages, train_samples, val_samples, test_samples):
    train_splits, val_splits, test_splits = [], [], []
    

    for lang in languages:
        print(f"Loading sentiment — language: {lang}")
        ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", lang,trust_remote_code=True)

        def subsample(split, n):
            return ds[split].shuffle(42).select(range(min(n, len(ds[split])))) if n else ds[split]

        ds_train = subsample("train",      train_samples)
        ds_val   = subsample("validation", val_samples)
        ds_test  = subsample("test",       test_samples)

        train_splits.append(ds_train)
        val_splits.append(ds_val)
        test_splits.append(ds_test)
        print(f"  train: {len(ds_train)} | val: {len(ds_val)} | test: {len(ds_test)}")
        merged_data=DatasetDict({
        "train":      concatenate_datasets(train_splits),
        "validation": concatenate_datasets(val_splits),
        "test":       concatenate_datasets(test_splits),
        })
        RAW_DIR.mkdir(parents=True,exist_ok=True)
        
        print(f"💾 Caching immutable raw data assets to: {RAW_DIR}")
        merged_data.save_to_disk(str(RAW_DIR))
        print(f'Sentiment Raw Data Saved Local in {RAW_DIR}')

    return merged_data


