# Multilingual NER & Sentiment Analysis System

[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-served-teal)]()
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)]()
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)]()
[![Docker](https://img.shields.io/badge/Docker-containerized-blue)]()

An end-to-end NLP system that fine-tunes a single multilingual transformer backbone for two tasks — **Named Entity Recognition** and **Sentiment Analysis** — across **English, Arabic, German, and French**, and serves both models through a **FastAPI** inference API with a **Gradio** web UI, fully containerized with **Docker**.

<p align="center">
  <img src="demo.gif" width="1500" alt="Demo of the Multilingual NER and Sentiment Analysis UI"/>
</p>

---

## Overview

Most NER/sentiment demos are trained and evaluated on English-only data. This project instead fine-tunes **XLM-RoBERTa** — a transformer pretrained on 100 languages — on multilingual datasets so the same backbone architecture learns to extract entities and classify sentiment across four languages at once, without maintaining separate models per language.

The project covers the full lifecycle of a real ML product:
- Data acquisition, subsampling, and multilingual dataset merging
- Subword tokenization with word-to-subtoken label alignment (the trickiest part of token classification)
- Fine-tuning with Hugging Face `Trainer`, mixed precision, and per-class metrics
- Model versioning and hosting on the Hugging Face Hub
- Production-style serving via a FastAPI inference layer with a Pydantic-validated config system
- An interactive Gradio front end for non-technical users
- Multi-stage Docker builds and Docker Compose orchestration for reproducible deployment

## Demo

The GIF above shows the Gradio UI accepting text in any of the four supported languages and returning both extracted entities and a sentiment label in real time, backed by the FastAPI inference service running behind it.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Gradio UI (7860)  │   ← user-facing web app
                    └──────────┬──────────┘
                               │ HTTP
                    ┌──────────▼──────────┐
                    │  FastAPI API (8000) │   ← inference service
                    │  /ner  /sentiment   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴───────────────┐
                 │                             │
        ┌────────▼─────────┐         ┌─────────▼─────────┐
        │  NER pipeline    │         │ Sentiment pipeline│
        │  (XLM-RoBERTa)   │         │  (XLM-RoBERTa)    │
        └────────┬─────────┘         └────────┬──────────┘
                 │                            │
        local models/ or Hugging Face Hub fallback
```

The API and UI are two separate Docker services that communicate over an internal network — a deliberately microservice-style split rather than a monolith, so either component can be scaled, redeployed, or swapped independently.

---

## Skills & Techniques Demonstrated

**NLP / Modeling**
- Fine-tuning a pretrained multilingual transformer (`xlm-roberta-base`) for two distinct task heads: token classification (NER) and sequence classification (sentiment)
- Multilingual dataset construction — loading and concatenating per-language splits from WikiANN (NER) and CardiffNLP's multilingual tweet sentiment corpus
- Subword tokenization with **word-to-token label alignment**: assigning the correct label only to the first subtoken of each word and masking continuation subtokens and special tokens (`[CLS]`, `[SEP]`, `[PAD]`) with `-100` so they're ignored by the loss function
- Class-imbalance-aware evaluation — weighted F1 and per-class F1, not just accuracy
- Sequence-labeling-specific evaluation via `seqeval` (entity-level precision/recall/F1, not token-level)
- Mixed-precision (`fp16`) training with the Hugging Face `Trainer` API, `DataCollatorForTokenClassification`, and `DataCollatorWithPadding`
- Model versioning and publishing to the Hugging Face Hub

**Engineering / MLOps**
- Config management with **Pydantic** for validated, typed settings loaded from `config.yaml`
- Environment-based fallback pattern: API loads a **local fine-tuned checkpoint** if present, otherwise automatically falls back to pulling the model from the Hugging Face Hub — makes the project runnable out-of-the-box for anyone cloning the repo
- REST API design with **FastAPI**, including request/response schema validation and startup-time model loading via a `lifespan` handler (models loaded once, shared across requests — no per-request reload overhead)
- Interactive UI with **Gradio**, decoupled from the API via HTTP so either can be developed, tested, or deployed independently
- **Multi-stage Docker builds** — separate build and runtime layers to keep production images free of build tools and package caches
- **Dependency-group management with `uv`** — split into `api`, `ui`, and `train` optional dependency groups so each Docker image installs only what it actually needs (the UI image never pulls in `torch`/`transformers`, cutting image size significantly)
- **Docker Compose** orchestration with health checks, read-only volume mounts for model weights, and service dependency ordering
- Reproducible environments via `uv.lock`, avoiding "works on my machine" drift

---

## Datasets

| Task | Dataset | Languages | Labels |
|---|---|---|---|
| NER | [WikiANN](https://huggingface.co/datasets/unimelb-nlp/wikiann) | English, Arabic, German, French | `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC` |
| Sentiment | [CardiffNLP Tweet Sentiment Multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) | English, Arabic, German, French | `negative`, `neutral`, `positive` |

Per-language splits were loaded independently and concatenated into unified multilingual train/validation/test sets, so the model is trained to generalize across languages within a single forward pass rather than needing language-specific routing.

## Model

- **Base checkpoint:** `xlm-roberta-base`
- **Fine-tuned separately** for each task (two checkpoints, shared backbone architecture)
- **Training:** Hugging Face `Trainer`, mixed precision (`fp16`), linear warmup, weighted-F1 model selection (`load_best_model_at_end`)

### Results

**NER** (entity-level, via `seqeval`, on held-out multilingual test set):

| Metric | Score |
|---|---|
| Precision | 0.7711 |
| Recall | 0.8031 |
| F1 | 0.7868 |
| Accuracy | 0.9171 |

**Sentiment** (3-class: negative / neutral / positive, on held-out multilingual test set):

| Metric | Score |
|---|---|
| Accuracy | 0.7046 |
| F1 (weighted) | 0.7015 |
| F1 (negative) | 0.7491 |
| F1 (neutral) | 0.6221 |
| F1 (positive) | 0.7332 |

3-class sentiment on short, informal multilingual text is a harder task than binary sentiment — published baselines on this same dataset (CardiffNLP's multilingual tweet sentiment benchmark) typically fall in the 0.65–0.72 weighted F1 range, which this result is consistent with. The neutral class is the weakest across all published models on this benchmark, not just this one, since neutral sentiment is inherently more ambiguous to annotate and classify in short-form text.

### Impact of fine-tuning

To quantify what fine-tuning actually contributed, the same `xlm-roberta-base` backbone was evaluated on the same held-out test sets with **untrained, randomly initialized task heads** (i.e. the model before any task-specific training):

**NER:**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Baseline (no fine-tuning) | 0.0267 | 0.1568 | 0.0457 | 0.0565 |
| Fine-tuned (this project) | 0.7711 | 0.8031 | 0.7868 | 0.9171 |

**Sentiment:**

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline (no fine-tuning) | 0.3333 | 0.1667 |
| Fine-tuned (this project) | 0.6983 | 0.6933 |

With randomly initialized task heads, both models perform close to chance — NER especially, since a 7-way per-token classification problem gives an untrained head essentially no consistent signal to exploit. Fine-tuning takes entity-level F1 from 0.046 to 0.787 (a ~17x improvement) and sentiment weighted F1 from 0.167 to 0.693 (a ~4x improvement), directly demonstrating the value added by task-specific training on top of the pretrained multilingual backbone.

### Example predictions

**NER:**
```
Input: "Apple CEO Tim Cook announced a new partnership with Samsung in Seoul."
  [ORG] Apple
  [PER] Tim Cook
  [ORG] Samsung
  [LOC] Seoul
```

**Sentiment (multilingual):**
```
🟢 [English] "This product is absolutely amazing!" → positive (0.96)
🔴 [English] "Terrible service, I am extremely disappointed."  → negative (0.90)
🟢 [Arabic]  "هذا المنتج رائع جداً، أنصح به بشدة"                → positive (0.97)
🟢 [German]  "Das Produkt ist absolut fantastisch!"            → positive (0.97)
```

---

## Project Structure

```
├─ api/                  # FastAPI inference service
│  ├─ main.py            # app entrypoint, startup model loading (lifespan)
│  ├─ model_loader.py    # loads local checkpoints or falls back to HF Hub
│  ├─ predictor.py       # inference logic
│  └─ schemas.py         # Pydantic request/response models
├─ configs/               # Pydantic settings validated from config.yaml
├─ ui/
│  └─ app.py             # Gradio front end, calls the API over HTTP
├─ NLP_SRC/
│  ├─ NER/               # data loading, preprocessing, training, evaluation
│  └─ Sentiment/         # data loading, preprocessing, training, evaluation
├─ models/                # fine-tuned checkpoints (weights hosted on HF Hub)
├─ data/                  # raw & processed datasets (HF `datasets` cache format)
├─ Dockerfile.api
├─ Dockerfile.ui
├─ compose.yaml
├─ pyproject.toml / uv.lock
└─ config.yaml
```

---

## Running Locally

```bash
# install dependencies (base + api + ui + train extras)
uv sync --all-extras

# terminal 1 — start the API
uv run uvicorn api.main:app --reload --port 8000

# terminal 2 — start the UI
uv run python ui/app.py
```

API docs: `http://localhost:8000/docs`
Gradio UI: `http://localhost:7860`

## Running with Docker

```bash
docker compose up --build
```

This builds and starts both services — the API downloads/loads the fine-tuned models (locally if present under `models/`, otherwise from the Hugging Face Hub automatically) and the UI connects to it over the internal Docker network.

---

## Tech Stack

`Python 3.12` · `PyTorch` · `Transformers` · `Datasets` · `Accelerate` · `Evaluate` · `seqeval` · `FastAPI` · `Pydantic` · `Gradio` · `Docker` · `Docker Compose` · `uv`

---

## Author

Built as an end-to-end portfolio project covering multilingual NLP fine-tuning, model serving, and containerized deployment.