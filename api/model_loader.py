# ============================================================
# api/model_loader.py
# Loads both models ONCE on startup and keeps them in memory.
# Every request reuses the same loaded pipeline — no cold start.
# ============================================================

import os
from pathlib import Path

import torch
from transformers import pipeline

from configs.core import config
# 0 = first GPU, -1 = CPU
DEVICE = 0 if torch.cuda.is_available() else -1
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
print('project root :',PROJECT_ROOT)
# ── Model sources ─────────────────────────────────────────────
# Priority: local folder → HuggingFace Hub
# Set these to your HF Hub repo names as fallback

# NER_MODEL_PATH  = os.getenv("NER_MODEL_PATH",  "./models/ner-final")
# SENT_MODEL_PATH = os.getenv("SENT_MODEL_PATH", "./models/sentiment-final")

# NER_MODEL_PATH=r"E:\Multilingual NER and Sentiment Analysis System\models\ner\final"
NER_MODEL_PATH =PROJECT_ROOT / config.ner_model_settings.output_dir
# print('NER_MODEL_PATH',NER_MODEL_PATH)
# SENT_MODEL_PATH = r"E:\Multilingual NER and Sentiment Analysis System\models\sentiment\final"
SENT_MODEL_PATH =PROJECT_ROOT / config.sentiment_model_settings.output_dir
# print('SENT_MODEL_PATH',SENT_MODEL_PATH)


NER_HUB_FALLBACK  = os.getenv("NER_HUB_MODEL",  config.ner_model_settings.model_finetuned)
SENT_HUB_FALLBACK = os.getenv("SENT_HUB_MODEL", config.sentiment_model_settings.model_finetuned)


class ModelStore:
    """
    Singleton model store.
    Both pipelines are loaded once at app startup via the lifespan handler.
    All requests share the same instances — no per-request loading overhead.
    """
    _ner_pipe  = None
    _sent_pipe = None

    @classmethod
    def load_all(cls):
        """Called once at startup. Loads both models."""
        print("Loading NER model...")
        ner_source = str(NER_MODEL_PATH) if os.path.exists(NER_MODEL_PATH) else NER_HUB_FALLBACK
        cls._ner_pipe = pipeline(
            "ner",
            model=ner_source,
            tokenizer=ner_source,
            aggregation_strategy="first",
            device=DEVICE,
        )
        print(f"  NER loaded from: {ner_source}")

        print("Loading Sentiment model...")
        sent_source = str(SENT_MODEL_PATH) if os.path.exists(SENT_MODEL_PATH) else SENT_HUB_FALLBACK
        cls._sent_pipe = pipeline(
            "text-classification",
            model=sent_source,
            tokenizer=sent_source,
            device=DEVICE,
        )
        print(f"  Sentiment loaded from: {sent_source}")
        print("Both models ready.")

    @classmethod
    def get_ner(cls):
        if cls._ner_pipe is None:
            raise RuntimeError("NER model not loaded. Call ModelStore.load_all() first.")
        return cls._ner_pipe

    @classmethod
    def get_sentiment(cls):
        if cls._sent_pipe is None:
            raise RuntimeError("Sentiment model not loaded. Call ModelStore.load_all() first.")
        return cls._sent_pipe

    @classmethod
    def is_ready(cls):
        return cls._ner_pipe is not None and cls._sent_pipe is not None
