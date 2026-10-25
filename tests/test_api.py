"""
test_api.py: integration tests for backend/api.py (FastAPI endpoints).

Uses FastAPI's TestClient so no real server is needed. The detector and the
feedback writer are mocked so tests are fast and fully offline.
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


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import agent.graph as ag
    import backend.api as api_module

    with patch.object(ag.detector, "analyze", return_value=MOCK_ANALYSIS), \
         patch.object(ag.feedback_writer, "summarise", return_value="Teacher note."):
        yield TestClient(api_module.app)


class TestHealthCheck:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_names_writelens(self, client):
        assert "WriteLens" in client.get("/").json()["message"]


class TestAnalyzeEndpoint:
    def test_returns_200(self, client, AI_TEXT):
        assert client.post("/analyze", json={"text": AI_TEXT}).status_code == 200

    def test_response_fields(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        for key in ("perplexity", "burstiness", "ai_score", "signal_level",
                    "reasons", "sentences", "summary", "disclaimer"):
            assert key in data

    def test_sentences_are_flagged(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert [s["flagged"] for s in data["sentences"]] == [True, False]

    def test_signal_level(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert data["signal_level"] == "Strong AI-like signals"

    def test_summary_included_by_default(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert data["summary"] == "Teacher note."

    def test_summary_can_be_turned_off(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT, "include_feedback": False}).json()
        assert data["summary"] == ""

    def test_disclaimer_always_present(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert "mistakes" in data["disclaimer"].lower()


class TestValidation:
    def test_too_short_rejected(self, client):
        assert client.post("/analyze", json={"text": "hi"}).status_code == 422

    def test_too_long_rejected(self, client):
        assert client.post("/analyze", json={"text": "a" * 9000}).status_code == 422

    def test_missing_text_rejected(self, client):
        assert client.post("/analyze", json={}).status_code == 422


class TestNoRewriteEndpoint:
    def test_humanize_endpoint_removed(self, client, AI_TEXT):
        assert client.post("/humanize", json={"text": AI_TEXT}).status_code == 404
