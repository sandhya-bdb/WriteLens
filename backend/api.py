import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from agent.graph import app as agent_app
from logic.detector import MAX_CHARS as DETECTOR_MAX_CHARS

# LangSmith: zero effect if LANGCHAIN_TRACING_V2 is not set.
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="WriteLens API",
    version="2.0.0",
    description=(
        "Explainable AI-writing analysis for educators. Scores text with GPT-2 "
        "perplexity and burstiness, highlights the most predictable sentences and "
        "explains the result in plain language. Scores are signals, not proof."
    ),
)

_MAX_CHARS = DETECTOR_MAX_CHARS  # 8 000
_MIN_CHARS = 10


# ── Pydantic models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    include_feedback: bool = True

    @field_validator("text")
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < _MIN_CHARS:
            raise ValueError(
                f"Text is too short ({len(stripped)} chars). "
                f"Provide at least {_MIN_CHARS} characters."
            )
        if len(stripped) > _MAX_CHARS:
            raise ValueError(
                f"Text is too long ({len(stripped)} chars). "
                f"Maximum is {_MAX_CHARS} characters (~2 000 words)."
            )
        return stripped


class SentenceScore(BaseModel):
    text: str
    perplexity: float
    flagged: bool


class AnalysisResponse(BaseModel):
    perplexity: float
    burstiness: float
    ai_score: int
    signal_level: str
    reasons: list[str]
    sentences: list[SentenceScore]
    summary: str
    disclaimer: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "WriteLens API is running", "version": "2.0.0"}


@traceable(name="writelens_pipeline", run_type="chain", tags=["writelens", "api"])
def _run_pipeline(text: str, include_feedback: bool) -> dict:
    """Wrapper so LangSmith shows the whole analyze -> explain -> feedback run as one trace."""
    return agent_app.invoke({"text": text, "include_feedback": include_feedback})


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for AI-like writing patterns and explain why.
    Returns document scores, per-sentence scores with flags, reasons,
    an optional teacher-friendly summary and an honest-use disclaimer.
    """
    logger.info("/analyze: %d chars, feedback=%s", len(request.text), request.include_feedback)
    try:
        state = _run_pipeline(request.text, request.include_feedback)
        analysis, report = state["analysis"], state["report"]
        logger.info(
            "/analyze done: ppl=%.1f burst=%.1f score=%d level=%s",
            analysis["perplexity"], analysis["burstiness"], analysis["ai_score"], report["signal_level"],
        )
        return {
            "perplexity":   analysis["perplexity"],
            "burstiness":   analysis["burstiness"],
            "ai_score":     analysis["ai_score"],
            "signal_level": report["signal_level"],
            "reasons":      report["reasons"],
            "sentences":    report["sentences"],
            "summary":      state.get("summary", ""),
            "disclaimer":   report["disclaimer"],
        }
    except ValueError as exc:
        logger.warning("/analyze validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("/analyze unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
