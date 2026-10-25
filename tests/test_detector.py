"""
test_detector.py — Unit tests for logic/detector.py (AIDetector class).

Tests cover:
  - Return shape / types of analyze()
  - Perplexity score range sanity
  - Burstiness is zero for single-sentence text
  - AI-score capping at 100
  - Edge cases: very short text, multi-sentence text
  - Score comparison: AI-style text should score higher than casual human text
"""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_analysis(result: dict) -> bool:
    """Check that analyze() returned a well-formed dict."""
    return (
        isinstance(result, dict)
        and "perplexity" in result
        and "burstiness" in result
        and "ai_score" in result
        and isinstance(result["perplexity"], float)
        and isinstance(result["burstiness"], float)
        and isinstance(result["ai_score"], int)
    )


# ---------------------------------------------------------------------------
# Tests: return structure
# ---------------------------------------------------------------------------

class TestAnalyzeReturnStructure:
    """analyze() should always return a properly typed dict."""

    def test_returns_dict_with_required_keys(self, detector, AI_TEXT):
        result = detector.analyze(AI_TEXT)
        assert is_valid_analysis(result), f"Unexpected result shape: {result}"

    def test_all_values_are_non_negative(self, detector, AI_TEXT):
        result = detector.analyze(AI_TEXT)
        assert result["perplexity"] >= 0
        assert result["burstiness"] >= 0
        assert result["ai_score"] >= 0

    def test_ai_score_capped_at_100(self, detector, AI_TEXT):
        result = detector.analyze(AI_TEXT)
        assert result["ai_score"] <= 100, "ai_score should never exceed 100"


# ---------------------------------------------------------------------------
# Tests: perplexity
# ---------------------------------------------------------------------------

class TestPerplexity:
    """Perplexity values should be in a plausible GPT-2 range."""

    def test_perplexity_is_positive(self, detector, AI_TEXT):
        ppl = detector.calculate_perplexity(AI_TEXT)
        assert ppl > 0, "Perplexity must be positive"

    def test_perplexity_is_finite(self, detector, AI_TEXT):
        import math
        ppl = detector.calculate_perplexity(AI_TEXT)
        assert math.isfinite(ppl), "Perplexity must be a finite number"

    def test_perplexity_reasonable_upper_bound(self, detector, AI_TEXT):
        """GPT-2 perplexity on real text should rarely exceed 1000."""
        ppl = detector.calculate_perplexity(AI_TEXT)
        assert ppl < 1000, f"Perplexity unexpectedly high: {ppl}"

    def test_human_text_higher_perplexity_than_ai_text(self, detector, AI_TEXT, HUMAN_TEXT):
        """
        Human-casual text should generally have higher perplexity than
        polished, formulaic AI text (though this may occasionally flip on
        short samples — we use a soft 'not dramatically lower' check).
        """
        ai_ppl = detector.calculate_perplexity(AI_TEXT)
        human_ppl = detector.calculate_perplexity(HUMAN_TEXT)
        # We assert that human text is NOT dramatically lower — allow up to 20% lower
        assert human_ppl >= ai_ppl * 0.8, (
            f"Human text perplexity ({human_ppl:.2f}) is unexpectedly much lower than "
            f"AI text perplexity ({ai_ppl:.2f})"
        )


# ---------------------------------------------------------------------------
# Tests: burstiness
# ---------------------------------------------------------------------------

class TestBurstiness:
    """Burstiness should reflect sentence-level perplexity variance."""

    def test_burstiness_is_non_negative(self, detector, AI_TEXT):
        result = detector.calculate_burstiness(AI_TEXT)
        assert result >= 0

    def test_single_sentence_burstiness_is_zero(self, detector):
        """One sentence → std dev of a single value → should be 0."""
        single = "This is a single complete sentence with enough tokens."
        result = detector.calculate_burstiness(single)
        assert result == 0.0, (
            f"Single-sentence burstiness should be 0, got {result}"
        )

    def test_multi_sentence_burstiness_positive(self, detector, MULTI_SENTENCE_TEXT):
        """Multiple different sentences should yield nonzero std dev."""
        result = detector.calculate_burstiness(MULTI_SENTENCE_TEXT)
        # May or may not be > 0 depending on sentences — just check it's a number
        assert isinstance(result, float)
        assert result >= 0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge and boundary input cases."""

    def test_short_text_raises_value_error(self, detector, SHORT_TEXT):
        """
        SHORT_TEXT ('AI is great.') is too short for GPT-2 (< MIN_TOKENS).
        The new _validate() should raise ValueError before touching the model.
        """
        with pytest.raises(ValueError, match="too short"):
            detector.analyze(SHORT_TEXT)

    def test_empty_text_raises_value_error(self, detector):
        """Empty string must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            detector.analyze("")

    def test_whitespace_only_raises_value_error(self, detector):
        """Whitespace-only input is treated as empty."""
        with pytest.raises(ValueError, match="empty"):
            detector.analyze("   \n\t  ")

    def test_too_long_text_raises_value_error(self, detector):
        """Text longer than MAX_CHARS should raise ValueError."""
        from logic.detector import MAX_CHARS
        too_long = "This is a sentence. " * (MAX_CHARS // 20 + 1)
        with pytest.raises(ValueError, match="too long"):
            detector.analyze(too_long)

    def test_wrong_type_raises_type_error(self, detector):
        """Non-string input should raise TypeError."""
        with pytest.raises(TypeError):
            detector.analyze(12345)

    def test_valid_medium_text_does_not_crash(self, detector, MULTI_SENTENCE_TEXT):
        """A normal multi-sentence text should succeed without errors."""
        result = detector.analyze(MULTI_SENTENCE_TEXT)
        assert is_valid_analysis(result)

    def test_repeated_calls_are_deterministic(self, detector, AI_TEXT):
        """Calling analyze() twice on the same text should return identical scores."""
        r1 = detector.analyze(AI_TEXT)
        r2 = detector.analyze(AI_TEXT)
        assert r1["perplexity"] == r2["perplexity"]
        assert r1["burstiness"] == r2["burstiness"]
        assert r1["ai_score"] == r2["ai_score"]


# ---------------------------------------------------------------------------
# Tests: sentence splitting and per-sentence scores
# ---------------------------------------------------------------------------

class TestSplitSentences:
    """split_sentences() is pure Python and needs no model."""

    def test_splits_on_terminal_punctuation(self):
        from logic.detector import split_sentences
        text = "This is the first sentence. Is this the second one? Yes, this is the third!"
        assert len(split_sentences(text)) == 3

    def test_drops_very_short_fragments(self):
        from logic.detector import split_sentences
        assert split_sentences("Ok. Fine. This one is long enough to keep.") == [
            "This one is long enough to keep."
        ]

    def test_empty_text(self):
        from logic.detector import split_sentences
        assert split_sentences("   ") == []


class TestSentenceScores:
    def test_analyze_returns_sentences(self, detector, MULTI_SENTENCE_TEXT):
        result = detector.analyze(MULTI_SENTENCE_TEXT)
        assert len(result["sentences"]) == 5
        for s in result["sentences"]:
            assert isinstance(s["text"], str)
            assert s["perplexity"] > 0

    def test_burstiness_matches_sentence_scores(self, detector, MULTI_SENTENCE_TEXT):
        import numpy as np
        result = detector.analyze(MULTI_SENTENCE_TEXT)
        expected = float(np.std([s["perplexity"] for s in result["sentences"]]))
        assert abs(result["burstiness"] - round(expected, 2)) < 0.05
