"""
test_agent.py: tests for agent/graph.py (analyze -> explain -> feedback).

The detector and feedback writer are mocked, so no GPT-2 inference or
Groq calls happen.
"""
from unittest.mock import patch

import pytest

MOCK_ANALYSIS = {
    "perplexity": 22.0,
    "burstiness": 4.0,
    "ai_score": 90,
    "sentences": [
        {"text": "The algorithm processes data efficiently.", "perplexity": 15.0},
        {"text": "Honestly my cat broke the router again lol.", "perplexity": 140.0},
    ],
}


@pytest.fixture
def ag():
    import agent.graph as ag
    return ag


class TestNodes:
    def test_analyze_node_returns_analysis(self, ag):
        with patch.object(ag.detector, "analyze", return_value=MOCK_ANALYSIS):
            out = ag.analyze_node({"text": "anything"})
        assert out == {"analysis": MOCK_ANALYSIS}

    def test_explain_node_builds_report(self, ag):
        out = ag.explain_node({"analysis": MOCK_ANALYSIS})
        report = out["report"]
        assert report["signal_level"] == "Strong AI-like signals"
        assert [s["flagged"] for s in report["sentences"]] == [True, False]

    def test_feedback_node_skipped_when_disabled(self, ag):
        out = ag.feedback_node({"include_feedback": False, "analysis": MOCK_ANALYSIS, "report": {}})
        assert out == {"summary": ""}

    def test_feedback_node_calls_writer(self, ag):
        with patch.object(ag.feedback_writer, "summarise", return_value="note") as m:
            out = ag.feedback_node({"include_feedback": True, "analysis": MOCK_ANALYSIS, "report": {"x": 1}})
        assert out == {"summary": "note"}
        m.assert_called_once()


class TestGraph:
    def test_full_pipeline(self, ag):
        with patch.object(ag.detector, "analyze", return_value=MOCK_ANALYSIS), \
             patch.object(ag.feedback_writer, "summarise", return_value="note"):
            state = ag.app.invoke({"text": "anything", "include_feedback": True})
        assert state["analysis"] == MOCK_ANALYSIS
        assert state["report"]["signal_level"] == "Strong AI-like signals"
        assert state["summary"] == "note"

    def test_pipeline_has_no_rewrite_step(self, ag):
        """WriteLens only analyzes: the graph must never modify the input text."""
        assert "rewriter" not in ag.app.get_graph().nodes
        with patch.object(ag.detector, "analyze", return_value=MOCK_ANALYSIS), \
             patch.object(ag.feedback_writer, "summarise", return_value="note"):
            state = ag.app.invoke({"text": "original text", "include_feedback": True})
        assert state["text"] == "original text"
