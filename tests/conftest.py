"""
conftest.py — shared fixtures for the entire WriteLens test suite.
"""
import pytest


# ---------------------------------------------------------------------------
# Sample text constants (also available as fixtures)
# ---------------------------------------------------------------------------

_AI_TEXT = (
    "Artificial intelligence is revolutionizing numerous sectors of the modern economy. "
    "Furthermore, it is crucial to leverage advanced machine learning algorithms to optimize "
    "business outcomes. The utilization of AI-driven solutions enables organizations to "
    "achieve unprecedented levels of efficiency and innovation. Consequently, businesses "
    "that implement these transformative technologies showcase remarkable competitive advantages."
)

_HUMAN_TEXT = (
    "Honestly, I've been putting off learning to cook for years — it just always felt like "
    "too much work. But then my sister made this ridiculously good pasta last month, and I "
    "kind of had to admit she was onto something. So yeah... I've been at it for three weeks now. "
    "Some disasters, a few accidental wins. Yesterday's soup was actually pretty solid."
)

_SHORT_TEXT = "AI is great."

_MULTI_SENTENCE_TEXT = (
    "The algorithm processes data efficiently. "
    "Machine learning models are trained on large datasets. "
    "Neural networks learn complex patterns. "
    "Deep learning requires significant computational resources. "
    "The results demonstrate improved performance metrics."
)


# ---------------------------------------------------------------------------
# Fixtures — text samples
# ---------------------------------------------------------------------------

@pytest.fixture
def AI_TEXT():
    return _AI_TEXT


@pytest.fixture
def HUMAN_TEXT():
    return _HUMAN_TEXT


@pytest.fixture
def SHORT_TEXT():
    return _SHORT_TEXT


@pytest.fixture
def MULTI_SENTENCE_TEXT():
    return _MULTI_SENTENCE_TEXT


# ---------------------------------------------------------------------------
# Lazy singleton AIDetector — loaded once per session to avoid slow re-init
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def detector():
    """
    Returns a single AIDetector instance reused across the entire test session.
    GPT-2 is ~500 MB — we only want to load it once.
    """
    from logic.detector import AIDetector
    return AIDetector()
