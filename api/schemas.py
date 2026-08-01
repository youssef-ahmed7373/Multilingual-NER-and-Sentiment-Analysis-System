
# ============================================================
# api/schemas.py
# All request / response data shapes for the API
# ============================================================

from typing import List, Optional

from pydantic import BaseModel, Field

# ── Shared ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    models:  List[str]
    version: str


# ── NER ──────────────────────────────────────────────────────

class EntitySpan(BaseModel):
    text:  str
    label: str        # PER | ORG | LOC
    start: int
    end:   int
    score: float


# ── Sentiment ─────────────────────────────────────────────────

class SentimentResult(BaseModel):
    label: str        # positive | neutral | negative
    score: float      # confidence 0–1


# ── Combined predict ──────────────────────────────────────────

class PredictRequest(BaseModel):
    text:     str            = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = None   # auto-detected if None


class PredictResponse(BaseModel):
    text:          str
    language:      str
    entities:      List[EntitySpan]
    sentiment:     SentimentResult
    processing_ms: int


# ── Batch ─────────────────────────────────────────────────────

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)


class BatchItem(BaseModel):
    text:      str
    entities:  List[EntitySpan]
    sentiment: SentimentResult


class BatchResponse(BaseModel):
    results:       List[BatchItem]
    processing_ms: int
