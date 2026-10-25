import html
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Use BACKEND_URL env var so the frontend works in any deployment environment.
# Defaults to localhost for local development.
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="WriteLens | Explainable AI-Writing Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f18; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #12141f !important;
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] * { color: #c8cde8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #b0b7d4 !important; font-size: 0.88rem; line-height: 1.6; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #c850c0 100%);
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.5);
    color: #fff !important;
    border: none !important;
}

/* ── Text Areas ──────────────────────────────────────────── */
.stTextArea textarea {
    background: #1a1d2e !important;
    color: #e8eaf6 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}
.stTextArea textarea:focus { border-color: #6C63FF !important; box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important; }
.stTextArea label { color: #9098c8 !important; font-weight: 600 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Metric Cards ────────────────────────────────────────── */
.m-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}
.m-card.good  { border-top: 3px solid #22c55e; }
.m-card.warn  { border-top: 3px solid #f59e0b; }
.m-card.bad   { border-top: 3px solid #ef4444; }
.m-val { font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0; }
.good .m-val  { color: #22c55e; }
.warn .m-val  { color: #f59e0b; }
.bad  .m-val  { color: #ef4444; }
.m-name { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }
.m-sub  { font-size: 0.8rem; color: #9098c8; margin-top: 0.3rem; }

/* ── Verdict badge ───────────────────────────────────────── */
.badge-human { display:inline-block; background:rgba(34,197,94,0.15); color:#22c55e; border:1px solid #22c55e; padding:0.25rem 0.9rem; border-radius:2rem; font-weight:700; font-size:0.9rem; }
.badge-ai    { display:inline-block; background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid #ef4444; padding:0.25rem 0.9rem; border-radius:2rem; font-weight:700; font-size:0.9rem; }

/* ── Sentence view ───────────────────────────────────────── */
.sent-box { background:#1a1d2e; border:1px solid #2a2d3e; border-radius:12px; padding:1.2rem 1.4rem; line-height:1.85; font-size:0.92rem; color:#c8cde8; }

/* ── Section titles ──────────────────────────────────────────────────────────────────── */
.sec-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.10em; color:#6b7280; margin-bottom:0.6rem; }
hr.divider { border:none; border-top:1px solid #2a2d3e; margin:1.5rem 0; }

/* ── LangSmith badge ─────────────────────────────────────────────────────────────────── */
.ls-badge-on  { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(34,197,94,0.12); color:#22c55e; border:1px solid #22c55e; padding:0.22rem 0.75rem; border-radius:2rem; font-size:0.78rem; font-weight:700; }
.ls-badge-off { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(107,114,128,0.12); color:#6b7280; border:1px solid #374151; padding:0.22rem 0.75rem; border-radius:2rem; font-size:0.78rem; font-weight:600; }
.sent-flag { background:rgba(245,158,11,0.18); border-bottom:2px solid #f59e0b; border-radius:3px; padding:0 2px; }
.note-box { background:rgba(108,99,255,0.08); border:1px solid #3b3f63; border-radius:12px; padding:0.9rem 1.1rem; color:#c8cde8; font-size:0.88rem; line-height:1.6; }
.badge-mixed { display:inline-block; background:rgba(245,158,11,0.15); color:#f59e0b; border:1px solid #f59e0b; padding:0.25rem 0.9rem; border-radius:2rem; font-weight:700; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────
SAMPLE_TEXT = (
    "Artificial intelligence is fundamentally transforming the landscape of modern business operations. "
    "By leveraging advanced machine learning algorithms, organizations can now utilize data-driven insights "
    "to make more informed strategic decisions. Furthermore, the implementation of AI technologies enables "
    "companies to achieve unprecedented levels of operational efficiency. It is crucial for businesses to "
    "embrace these transformative technologies in order to remain competitive in today's rapidly evolving "
    "digital marketplace. Moreover, the integration of AI systems facilitates the automation of repetitive "
    "tasks, thereby allowing human employees to focus on more creative and high-value activities."
)


def ppl_cls(v):
    return "good" if v >= 60 else ("warn" if v >= 35 else "bad")


def burst_cls(v):
    return "good" if v >= 20 else ("warn" if v >= 8 else "bad")


def level_badge(level):
    cls = {"Strong AI-like signals": "badge-ai", "Mixed signals": "badge-mixed"}.get(level, "badge-human")
    return f'<span class="{cls}">{html.escape(level)}</span>'


def card(name, val, sub, cls):
    st.markdown(f"""
    <div class="m-card {cls}">
        <div class="m-name">{name}</div>
        <div class="m-val">{val}</div>
        <div class="m-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sentence_view(sentences):
    """Render sentences inline, highlighting the flagged (highly predictable) ones."""
    out = []
    for s in sentences:
        text = html.escape(s["text"])
        if s["flagged"]:
            out.append(f'<span class="sent-flag" title="perplexity {s["perplexity"]}">{text}</span>')
        else:
            out.append(f'<span title="perplexity {s["perplexity"]}">{text}</span>')
    return " ".join(out)


# ─── Sidebar ───────────────────────────────────────────────────────────────
_ls_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
_ls_key_set = bool(os.environ.get("LANGCHAIN_API_KEY", "").strip())
_ls_project = os.environ.get("LANGCHAIN_PROJECT", "WriteLens")
_ls_active = _ls_enabled and _ls_key_set

with st.sidebar:
    st.markdown("## 🔍 WriteLens")
    st.markdown("---")
    st.markdown("🟢 **Backend:** FastAPI + LangGraph")
    st.markdown("🟢 **Scoring:** GPT-2 perplexity & burstiness")
    st.markdown("🟢 **Summary:** Groq Llama 3.3 70B (optional)")

    st.markdown("")
    st.markdown("**LangSmith Tracing**")
    if _ls_active:
        ls_url = f"https://smith.langchain.com/o/~/projects/p/{_ls_project}"
        st.markdown(
            f'<a href="{ls_url}" target="_blank">'
            f'<span class="ls-badge-on">● Active: {_ls_project}</span></a>',
            unsafe_allow_html=True
        )
    elif _ls_enabled and not _ls_key_set:
        st.markdown('<span class="ls-badge-off">⚠ Key missing</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="ls-badge-off">○ Disabled</span>', unsafe_allow_html=True)

    st.markdown("---")
    include_feedback = st.toggle("Teacher-friendly summary", value=True)

    st.markdown("---")
    st.markdown("### 📖 How to read the scores")
    st.markdown("""
**Perplexity:** how predictable the wording is to GPT-2.
Lower = more predictable, which is common in AI text.

**Burstiness:** how much sentence complexity varies.
Lower = very uniform sentences.

**Highlighted sentences** have highly predictable wording.

**Remember:** these are statistical signals, not proof of who wrote the text.
    """)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("## 🔍 WriteLens: Explainable AI-Writing Analysis")
st.markdown(
    "Paste a piece of writing to see **which patterns look AI-generated and why**, "
    "sentence by sentence. Built for teachers who want to understand a result, not just get a number."
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Input ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Text to analyze</div>', unsafe_allow_html=True)
if st.button("Load sample text"):
    st.session_state["sample_loaded"] = True

default_val = SAMPLE_TEXT if st.session_state.get("sample_loaded") else ""
input_text = st.text_area("Text", value=default_val, height=220,
                          placeholder="Paste text here (at least 3 sentences works best)…",
                          key="input_area", label_visibility="collapsed")

analyze_btn = st.button("🔍 Analyze writing")

if analyze_btn:
    if input_text.strip():
        with st.spinner("Scoring with GPT-2 and preparing the explanation…"):
            try:
                r = requests.post(
                    f"{API_URL}/analyze",
                    json={"text": input_text, "include_feedback": include_feedback},
                    timeout=120,
                )
                if r.status_code == 200:
                    st.session_state["report"] = r.json()
                else:
                    st.error(f"Backend error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Run: `uvicorn backend.api:app --reload`")
            except requests.exceptions.Timeout:
                st.error("⏱ Timed out. Try a shorter text.")
    else:
        st.warning("Paste some text first.")

# ─── Report ────────────────────────────────────────────────────────────────
if "report" in st.session_state:
    d = st.session_state["report"]
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Analysis &nbsp;&nbsp; {level_badge(d['signal_level'])}", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Perplexity", round(d["perplexity"], 1),
             "Lower = more predictable wording", ppl_cls(d["perplexity"]))
    with c2:
        card("Burstiness", round(d["burstiness"], 1),
             "Lower = more uniform sentences", burst_cls(d["burstiness"]))
    with c3:
        flagged = sum(1 for s in d["sentences"] if s["flagged"])
        card("Predictable sentences", f"{flagged} / {len(d['sentences'])}",
             "Highlighted below", "bad" if d["sentences"] and flagged / len(d["sentences"]) > 0.5 else "warn")

    st.markdown("### 🧾 Why this result")
    for reason in d["reasons"]:
        st.markdown(f"- {reason}")

    if d["sentences"]:
        st.markdown("### ✏️ Sentence by sentence")
        st.caption("Highlighted = highly predictable wording. Hover a sentence to see its perplexity.")
        st.markdown(f'<div class="sent-box">{sentence_view(d["sentences"])}</div>', unsafe_allow_html=True)

    if d.get("summary"):
        st.markdown("### 🧑‍🏫 Summary for teachers")
        st.markdown(f'<div class="note-box">{html.escape(d["summary"])}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.info(d["disclaimer"])

    with st.expander("🛠 Raw API response"):
        st.json(d)
