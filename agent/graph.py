"""
graph.py: the WriteLens analysis pipeline as a LangGraph.

    analyze  ->  explain  ->  feedback  ->  END

analyze   GPT-2 perplexity, burstiness and per-sentence scores
explain   plain-language reasons, flagged sentences, signal level
feedback  short teacher-friendly summary (Groq if configured, template otherwise)

Each node is traced in LangSmith when LANGCHAIN_TRACING_V2=true.
"""
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from logic.detector import AIDetector
from logic.explainer import explain
from logic.feedback import FeedbackWriter

load_dotenv()

# LangSmith: @traceable makes each function a named span in the trace.
# If langsmith is not installed this falls back to a no-op decorator.
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        """Fallback no-op decorator when langsmith is not installed."""
        def decorator(fn):
            return fn
        return decorator


# 1. State
class AgentState(TypedDict, total=False):
    text: str
    include_feedback: bool
    analysis: dict
    report: dict
    summary: str


# 2. Logic (loaded once; GPT-2 is ~500 MB)
detector = AIDetector()
feedback_writer = FeedbackWriter()


# 3. Nodes
@traceable(name="analyze_node", run_type="chain", tags=["writelens", "analyze"])
def analyze_node(state: AgentState) -> dict:
    return {"analysis": detector.analyze(state["text"])}


@traceable(name="explain_node", run_type="chain", tags=["writelens", "explain"])
def explain_node(state: AgentState) -> dict:
    return {"report": explain(state["analysis"])}


@traceable(name="feedback_node", run_type="chain", tags=["writelens", "feedback"])
def feedback_node(state: AgentState) -> dict:
    if not state.get("include_feedback", True):
        return {"summary": ""}
    return {"summary": feedback_writer.summarise(state["analysis"], state["report"])}


# 4. Graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("explain", explain_node)
workflow.add_node("feedback", feedback_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "explain")
workflow.add_edge("explain", "feedback")
workflow.add_edge("feedback", END)

app = workflow.compile()
