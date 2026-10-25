"""
explainer.py: turns raw detector numbers into plain-language explanations.

Pure Python (no torch), so it is cheap to test and easy to read.
The goal is to explain *why* a piece of writing shows AI-like patterns,
not to deliver a verdict about the person who wrote it.
"""

# Sentence-level threshold: GPT-2 perplexity below this means the wording
# is very predictable for a language model.
PREDICTABLE_SENTENCE_PPL = 30.0

# Document-level thresholds (same ones the ai_score heuristic uses)
LOW_PPL = 30.0
MID_PPL = 50.0
LOW_BURSTINESS = 10.0

DISCLAIMER = (
    "These scores describe statistical patterns in the writing, not who wrote it. "
    "AI-text detectors, including this one, make mistakes, and they are known to "
    "flag some human writers more often (for example non-native English speakers "
    "and very formal writing). Use the results to start a conversation, never as "
    "the only evidence for a decision about a student."
)


# From the HC3 evaluation (eval/results.json): 132 of 200 human answers
# landed in "Mixed signals", so this level alone says very little.
MIXED_NOTE = (
    "Mixed signals are common in ordinary human writing: in our HC3 test, "
    "about two in three human-written answers landed here. On its own this "
    "result is not a reason for concern."
)


def signal_level(ai_score: int) -> str:
    """Map the 0-100 heuristic score to a cautious, human-readable level."""
    if ai_score >= 70:
        return "Strong AI-like signals"
    if ai_score >= 40:
        return "Mixed signals"
    return "Few AI-like signals"


def flag_sentences(sentences: list[dict], threshold: float = PREDICTABLE_SENTENCE_PPL) -> list[dict]:
    """
    Mark sentences whose wording is highly predictable.

    Input:  [{"text": str, "perplexity": float}, ...]
    Output: same items plus "flagged": bool
    """
    return [{**s, "flagged": s["perplexity"] < threshold} for s in sentences]


def build_reasons(perplexity: float, burstiness: float, sentences: list[dict]) -> list[str]:
    """Plain-language reasons behind the score, most important first."""
    reasons = []

    if perplexity < LOW_PPL:
        reasons.append(
            f"Very predictable wording overall (perplexity {perplexity:.1f}). "
            "A language model can guess most of the next words, which is common in AI-generated text."
        )
    elif perplexity < MID_PPL:
        reasons.append(
            f"Fairly predictable wording (perplexity {perplexity:.1f}). "
            "This is borderline: polished human writing can also land here."
        )
    else:
        reasons.append(
            f"Less predictable wording (perplexity {perplexity:.1f}), which is more typical of human writing."
        )

    if len(sentences) < 3:
        reasons.append(
            "Too few sentences to judge variety between sentences, so treat the result with extra caution."
        )
    elif burstiness < LOW_BURSTINESS:
        reasons.append(
            f"Sentences are very uniform in complexity (burstiness {burstiness:.1f}). "
            "People usually mix simple and complex sentences more."
        )
    else:
        reasons.append(
            f"Sentence complexity varies (burstiness {burstiness:.1f}), which is more typical of human writing."
        )

    flagged = [s for s in sentences if s.get("flagged")]
    if sentences:
        reasons.append(
            f"{len(flagged)} of {len(sentences)} sentences have highly predictable wording."
        )

    return reasons


def explain(analysis: dict) -> dict:
    """
    Combine a detector.analyze() result into an explained report.

    Returns:
        {
            "signal_level": str,
            "reasons":      [str, ...],
            "sentences":    [{"text", "perplexity", "flagged"}, ...],
            "disclaimer":   str,
        }
    """
    sentences = flag_sentences(analysis.get("sentences", []))
    level = signal_level(analysis["ai_score"])
    reasons = build_reasons(analysis["perplexity"], analysis["burstiness"], sentences)
    if level == "Mixed signals":
        reasons.append(MIXED_NOTE)
    return {
        "signal_level": level,
        "reasons": reasons,
        "sentences": sentences,
        "disclaimer": DISCLAIMER,
    }
