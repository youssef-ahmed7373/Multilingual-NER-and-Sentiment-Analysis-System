# ============================================================
# api/main.py
# Run locally:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs at:      http://localhost:8000/docs
# ============================================================

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from langdetect import LangDetectException, detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from api.model_loader import ModelStore
from api.predictor import (run_batch_ner, run_batch_sentiment, run_ner,
                           run_sentiment)
from api.schemas import (BatchItem, BatchRequest, BatchResponse,
                         HealthResponse, PredictRequest, PredictResponse)


# ── App lifespan — load models on startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelStore.load_all()    # runs once when the server starts
    yield
    # nothing to clean up on shutdown

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual NER + Sentiment API",
    description="XLM-RoBERTa fine-tuned on WikiANN (NER) and Cardiff Twitter (Sentiment). Supports EN, AR, DE, FR.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict to your Gradio URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Multilingual NER + Sentiment API. Visit /docs for the interactive docs."}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """Check that the API is running and both models are loaded."""
    return HealthResponse(
        status="ok" if ModelStore.is_ready() else "loading",
        models=["xlmr-ner-multilingual", "xlmr-sentiment-multilingual"],
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Run NER + Sentiment on a single text.
    Language is auto-detected if not provided.
    """
    if not ModelStore.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading. Try again in a moment.")

    start = time.time()

    # Language detection
    if req.language:
        lang = req.language
    elif LANGDETECT_AVAILABLE:
        try:
            lang = detect(req.text)
        except LangDetectException:
            lang = "unknown"
    else:
        lang = "unknown"

    entities  = run_ner(req.text)
    sentiment = run_sentiment(req.text)
    ms        = int((time.time() - start) * 1000)

    return PredictResponse(
        text=req.text,
        language=lang,
        entities=entities,
        sentiment=sentiment,
        processing_ms=ms,
    )


@app.post("/batch", response_model=BatchResponse, tags=["Inference"])
def batch_predict(req: BatchRequest):
    """
    Run NER + Sentiment on up to 32 texts at once.
    Much faster than calling /predict in a loop.
    """
    if not ModelStore.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading.")

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list is empty.")

    start         = time.time()
    all_entities  = run_batch_ner(req.texts)
    all_sentiment = run_batch_sentiment(req.texts)
    ms            = int((time.time() - start) * 1000)

    results = [
        BatchItem(text=text, entities=ents, sentiment=sent)
        for text, ents, sent in zip(req.texts, all_entities, all_sentiment)
    ]
    return BatchResponse(results=results, processing_ms=ms)
