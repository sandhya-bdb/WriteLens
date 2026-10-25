"""
feedback.py: optional short, teacher-friendly summary of an analysis report.

Uses Groq (Llama 3.3 70B) when GROQ_API_KEY is set. The LLM only sees the
scores, reasons and flagged sentences, and is told to describe patterns,
not to accuse anyone and not to rewrite the text.

Without a key (or if the call fails) a deterministic template summary is
returned, so the app works fully offline.
"""
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You help teachers read the output of an AI-writing analysis tool.
You will get statistical scores, plain-language reasons and the sentences with the
most predictable wording.

Write 3 to 4 short sentences for a teacher:
1. Summarise what the patterns suggest, using cautious language ("shows patterns
   common in AI-generated text", never "was written by AI").
2. Point to which parts of the writing drove the result.
3. Suggest one fair next step, such as asking the student to talk through their
   drafting process or share earlier drafts.

Rules: do not accuse the student, do not claim certainty, do not rewrite or
improve the student's text, and do not invent scores."""


def template_summary(report: dict) -> str:
    """Deterministic fallback summary built only from the report."""
    flagged = [s for s in report.get("sentences", []) if s.get("flagged")]
    total = len(report.get("sentences", []))
    level = report["signal_level"]
    parts = [f"Result: {level.lower()}."]
    if total:
        parts.append(f"{len(flagged)} of {total} sentences use highly predictable wording.")
    if level == "Few AI-like signals":
        parts.append("Nothing here needs follow-up on its own.")
    else:
        parts.append(
            "If this matters for a decision, a fair next step is to ask the student "
            "to talk through how they drafted it or to share earlier versions."
        )
    return " ".join(parts)


class FeedbackWriter:
    def __init__(self, api_key: str | None = None):
        key = (api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")).strip()
        self.client = None
        if key:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                logger.info("FeedbackWriter: Groq client ready (%s).", MODEL)
            except Exception as exc:
                logger.exception("FeedbackWriter: could not initialise Groq: %s", exc)
        else:
            logger.info("FeedbackWriter: GROQ_API_KEY not set, using template summaries.")

    def summarise(self, analysis: dict, report: dict) -> str:
        if not self.client:
            return template_summary(report)

        flagged = [s["text"] for s in report.get("sentences", []) if s.get("flagged")][:5]
        user_msg = (
            f"Signal level: {report['signal_level']}\n"
            f"Heuristic score (0-100): {analysis['ai_score']}\n"
            f"Perplexity: {analysis['perplexity']}\n"
            f"Burstiness: {analysis['burstiness']}\n"
            "Reasons:\n- " + "\n- ".join(report["reasons"]) + "\n"
            "Most predictable sentences:\n- " + ("\n- ".join(flagged) if flagged else "(none)")
        )
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.exception("FeedbackWriter: Groq call failed, using template: %s", exc)
            return template_summary(report)
