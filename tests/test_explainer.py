"""
test_explainer.py: unit tests for logic/explainer.py (pure Python, no GPT-2).
"""
from logic.explainer import (
    DISCLAIMER,
    MIXED_NOTE,
    PREDICTABLE_SENTENCE_PPL,
    build_reasons,
    explain,
    flag_sentences,
    signal_level,
)


def _sentences(*ppls):
    return [{"text": f"Sentence number {i} is here.", "perplexity": p} for i, p in enumerate(ppls)]


class TestSignalLevel:
    def test_high_score_is_strong(self):
        assert signal_level(90) == "Strong AI-like signals"

    def test_boundary_70_is_strong(self):
        assert signal_level(70) == "Strong AI-like signals"

    def test_middle_score_is_mixed(self):
        assert signal_level(40) == "Mixed signals"

    def test_low_score_is_few(self):
        assert signal_level(10) == "Few AI-like signals"

    def test_never_claims_certainty(self):
        for score in range(0, 101, 10):
            level = signal_level(score).lower()
            assert "written by" not in level
            assert "definitely" not in level


class TestFlagSentences:
    def test_low_perplexity_is_flagged(self):
        out = flag_sentences(_sentences(PREDICTABLE_SENTENCE_PPL - 1))
        assert out[0]["flagged"] is True

    def test_high_perplexity_is_not_flagged(self):
        out = flag_sentences(_sentences(PREDICTABLE_SENTENCE_PPL + 50))
        assert out[0]["flagged"] is False

    def test_keeps_text_and_order(self):
        src = _sentences(10, 90, 20)
        out = flag_sentences(src)
        assert [s["text"] for s in out] == [s["text"] for s in src]

    def test_does_not_mutate_input(self):
        src = _sentences(10)
        flag_sentences(src)
        assert "flagged" not in src[0]

    def test_empty_input(self):
        assert flag_sentences([]) == []


class TestBuildReasons:
    def test_low_perplexity_mentions_predictable(self):
        reasons = build_reasons(20.0, 15.0, flag_sentences(_sentences(20, 25, 30)))
        assert "predictable" in reasons[0].lower()

    def test_high_perplexity_mentions_human(self):
        reasons = build_reasons(80.0, 25.0, flag_sentences(_sentences(80, 60, 120)))
        assert "human" in reasons[0].lower()

    def test_uniform_sentences_reason(self):
        reasons = build_reasons(40.0, 5.0, flag_sentences(_sentences(40, 42, 41)))
        assert any("uniform" in r.lower() for r in reasons)

    def test_few_sentences_adds_caution(self):
        reasons = build_reasons(40.0, 0.0, flag_sentences(_sentences(40)))
        assert any("caution" in r.lower() for r in reasons)

    def test_counts_flagged_sentences(self):
        reasons = build_reasons(25.0, 5.0, flag_sentences(_sentences(10, 20, 90)))
        assert any("2 of 3 sentences" in r for r in reasons)


class TestExplain:
    def test_report_shape(self):
        report = explain({"perplexity": 25.0, "burstiness": 5.0, "ai_score": 90,
                          "sentences": _sentences(10, 20, 90)})
        assert set(report) == {"signal_level", "reasons", "sentences", "disclaimer"}
        assert report["disclaimer"] == DISCLAIMER
        assert all("flagged" in s for s in report["sentences"])

    def test_missing_sentences_key_is_handled(self):
        report = explain({"perplexity": 80.0, "burstiness": 25.0, "ai_score": 10})
        assert report["sentences"] == []
        assert report["signal_level"] == "Few AI-like signals"

    def test_disclaimer_warns_about_mistakes(self):
        assert "mistakes" in DISCLAIMER.lower()
        assert "never" in DISCLAIMER.lower()


class TestMixedNote:
    def test_mixed_level_adds_caution_note(self):
        report = explain({"perplexity": 40.0, "burstiness": 15.0, "ai_score": 40,
                          "sentences": _sentences(40, 60, 35)})
        assert report["signal_level"] == "Mixed signals"
        assert MIXED_NOTE in report["reasons"]

    def test_strong_level_has_no_mixed_note(self):
        report = explain({"perplexity": 20.0, "burstiness": 5.0, "ai_score": 90,
                          "sentences": _sentences(20, 22, 21)})
        assert MIXED_NOTE not in report["reasons"]
