"""
test_feedback.py: tests for logic/feedback.py. No network calls.
"""
from unittest.mock import MagicMock

from logic.feedback import FeedbackWriter, template_summary

REPORT = {
    "signal_level": "Strong AI-like signals",
    "reasons": ["Very predictable wording overall (perplexity 20.0)."],
    "sentences": [
        {"text": "First sentence here.", "perplexity": 12.0, "flagged": True},
        {"text": "Second sentence here.", "perplexity": 80.0, "flagged": False},
    ],
    "disclaimer": "x",
}
ANALYSIS = {"perplexity": 20.0, "burstiness": 5.0, "ai_score": 90}


class TestTemplateSummary:
    def test_mentions_level_and_counts(self):
        text = template_summary(REPORT)
        assert "strong ai-like signals" in text.lower()
        assert "1 of 2 sentences" in text

    def test_suggests_fair_next_step_when_signals(self):
        assert "drafted" in template_summary(REPORT)

    def test_low_signal_needs_no_follow_up(self):
        low = {**REPORT, "signal_level": "Few AI-like signals"}
        assert "no" in template_summary(low).lower() or "nothing" in template_summary(low).lower()


class TestFeedbackWriter:
    def test_no_key_uses_template(self):
        writer = FeedbackWriter(api_key="")
        assert writer.client is None
        assert writer.summarise(ANALYSIS, REPORT) == template_summary(REPORT)

    def test_llm_response_is_returned(self):
        writer = FeedbackWriter(api_key="")
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="  LLM summary.  "))
        ]
        writer.client = fake
        assert writer.summarise(ANALYSIS, REPORT) == "LLM summary."

    def test_prompt_contains_flagged_sentence_only(self):
        writer = FeedbackWriter(api_key="")
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="ok"))]
        writer.client = fake
        writer.summarise(ANALYSIS, REPORT)
        user_msg = fake.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "First sentence here." in user_msg
        assert "Second sentence here." not in user_msg

    def test_llm_failure_falls_back_to_template(self):
        writer = FeedbackWriter(api_key="")
        fake = MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("network down")
        writer.client = fake
        assert writer.summarise(ANALYSIS, REPORT) == template_summary(REPORT)
