
# ============================================================
# api/predictor.py
# Pure inference logic — no FastAPI imports here.
# Keeps the business logic separate from the web layer.
# ============================================================

from api.model_loader import ModelStore
from api.schemas import EntitySpan, SentimentResult


def run_ner(text: str) -> list[EntitySpan]:
    """Run NER pipeline and return clean EntitySpan list."""
    ner_pipe = ModelStore.get_ner()
    raw      = ner_pipe(text)
    return [
        EntitySpan(
            text=e["word"],
            label=e["entity_group"],
            start=e["start"],
            end=e["end"],
            score=round(float(e["score"]), 4),
        )
        for e in raw
    ]


def run_sentiment(text: str, max_length: int = 128) -> SentimentResult:
    """Run sentiment pipeline and return clean SentimentResult."""
    sent_pipe = ModelStore.get_sentiment()
    result    = sent_pipe(text, truncation=True, max_length=max_length)[0]
    return SentimentResult(
        label=result["label"],
        score=round(float(result["score"]), 4),
    )


def run_batch_ner(texts: list[str]) -> list[list[EntitySpan]]:
    """Batch NER — faster than calling run_ner() in a loop."""
    ner_pipe = ModelStore.get_ner()
    raw_batch = ner_pipe(texts)
    return [
        [
            EntitySpan(
                text=e["word"],
                label=e["entity_group"],
                start=e["start"],
                end=e["end"],
                score=round(float(e["score"]), 4),
            )
            for e in raw
        ]
        for raw in raw_batch
    ]


def run_batch_sentiment(texts: list[str], max_length: int = 128) -> list[SentimentResult]:
    """Batch sentiment — faster than calling run_sentiment() in a loop."""
    sent_pipe = ModelStore.get_sentiment()
    results   = sent_pipe(texts, truncation=True, max_length=max_length)
    return [
        SentimentResult(
            label=r["label"],
            score=round(float(r["score"]), 4),
        )
        for r in results
    ]
