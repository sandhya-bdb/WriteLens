"""
test_evaluate.py: tests for the pure helpers in scripts/evaluate.py (no GPT-2).
"""
from scripts.evaluate import build_samples, metrics, to_markdown, usable

LONG = "This is sentence one here. This is sentence two here. This is sentence three here."


class TestUsable:
    def test_needs_three_sentences(self):
        assert usable(LONG)
        assert not usable("Only one sentence that is long enough.")

    def test_rejects_empty(self):
        assert not usable("   ")


class TestBuildSamples:
    def test_balanced_classes(self):
        rows = [{"human_answers": [LONG], "chatgpt_answers": [LONG]} for _ in range(10)]
        samples = build_samples(rows, n=4, seed=1)
        labels = [y for _, y in samples]
        assert labels.count(0) == 4 and labels.count(1) == 4

    def test_skips_unusable_answers(self):
        rows = [{"human_answers": ["too short."], "chatgpt_answers": [LONG]}]
        samples = build_samples(rows, n=5, seed=1)
        assert [y for _, y in samples] == [1]


class TestMetrics:
    def test_perfect(self):
        m = metrics([0, 0, 1, 1], [10, 20, 90, 80], threshold=70)
        assert m["accuracy"] == 1.0 and m["false_positive_rate"] == 0.0

    def test_false_positive_rate(self):
        m = metrics([0, 0, 0, 0], [90, 10, 10, 10], threshold=70)
        assert m["false_positive_rate"] == 0.25

    def test_markdown_table(self):
        result = {
            "n_human": 2, "n_ai": 2, "seed": 1,
            "thresholds": {"strong (>=70)": metrics([0, 1], [10, 90], 70)},
            "signal_levels": {"human": {"Few AI-like signals": 1}, "chatgpt": {"Strong AI-like signals": 1}},
        }
        md = to_markdown(result)
        assert "| strong (>=70) | 100% |" in md
