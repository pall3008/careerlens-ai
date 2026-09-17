"""
CareerLens AI v2 — Hybrid RAG + NLP + MCP + Voice Interview
New in v2:
  - ChromaDB persistent vector store (replaces FAISS)
  - BM25 + semantic hybrid search with Reciprocal Rank Fusion
  - spaCy NLP skill extractor (Skills Map step)
  - Whisper voice transcription for interview answers
  - FastMCP server (run separately: python src/mcp_server.py)
"""

import streamlit as st
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import RAGEngine, extract_text_from_pdf
from src.llm_agent import (
    analyze_resume_vs_jd,
    generate_interview_questions,
    evaluate_answer,
    chat_with_resume,
    extract_job_search_query,
    audit_resume_sections,
    rewrite_resume_section,
)
from src.job_search import get_job_matches
from src.nlp_engine import analyze_skills   # NEW — spaCy skill extractor

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: #0f0d1a; }
    section[data-testid="stSidebar"] {
        background: #130f22;
        border-right: 1px solid rgba(139, 92, 246, 0.12);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid rgba(139, 92, 246, 0.25);
        color: #c4b5fd; font-weight: 500;
        border-radius: 10px; transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.5);
    }
    .block-container { padding: 2rem 2.5rem; max-width: 1200px; }
    .hero-wrap { padding: 0.5rem 0 2rem 0; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #a78bfa; border-radius: 999px;
        padding: 5px 16px; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem;
    }
    .hero-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #a78bfa;
        display: inline-block; animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .main-title {
        font-size: 3rem; font-weight: 800; line-height: 1.1;
        background: linear-gradient(110deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.6rem; letter-spacing: -0.02em;
    }
    .hero-sub { color: #6b7280; font-size: 1.05rem; font-weight: 400; max-width: 520px; }
    .hero-sub strong { color: #9ca3af; font-weight: 600; }
    .stepper-wrap {
        display: flex; align-items: center;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1rem 1.5rem; margin: 0 0 2rem 0; gap: 0;
    }
    .step-node {
        flex: 1; display: flex; flex-direction: column;
        align-items: center; position: relative; gap: 6px;
    }
    .step-node::after {
        content: ''; position: absolute; top: 14px;
        left: calc(50% + 18px); right: calc(-50% + 18px);
        height: 1px; background: rgba(255,255,255,0.08);
    }
    .step-node:last-child::after { display: none; }
    .step-node.done::after { background: linear-gradient(90deg, #8b5cf6, rgba(139,92,246,0.3)); }
    .step-circle {
        width: 30px; height: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700; z-index: 2;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1); color: #4b5563;
    }
    .step-node.done .step-circle { background: #7c3aed; border-color: #7c3aed; color: white; }
    .step-node.active .step-circle {
        background: rgba(124,58,237,0.15); border: 2px solid #8b5cf6; color: #c4b5fd;
        box-shadow: 0 0 0 4px rgba(139,92,246,0.1);
    }
    .step-label { font-size: 0.68rem; font-weight: 500; color: #374151; text-align: center; white-space: nowrap; }
    .step-node.active .step-label { color: #e9e4ff; font-weight: 700; }
    .step-node.done .step-label { color: #7c6fad; }
    .card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1.2rem; text-align: center;
    }
    .metric-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #4b5563; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 2.8rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
    .metric-sub { font-size: 0.78rem; color: #6b7280; margin-top: 4px; }
    .tag {
        display: inline-block; background: rgba(124,58,237,0.12);
        border: 1px solid rgba(139,92,246,0.28); border-radius: 999px;
        padding: 4px 14px; margin: 3px; font-size: 0.8rem; font-weight: 600; color: #c4b5fd;
    }
    .tag-green  { background:rgba(16,185,129,0.1);  border-color:rgba(16,185,129,0.25); color:#6ee7b7; }
    .tag-blue   { background:rgba(59,130,246,0.1);  border-color:rgba(59,130,246,0.25); color:#93c5fd; }
    .tag-red    { background:rgba(239,68,68,0.1);   border-color:rgba(239,68,68,0.25);  color:#fca5a5; }
    .tag-amber  { background:rgba(245,158,11,0.1);  border-color:rgba(245,158,11,0.25); color:#fcd34d; }
    .section-eyebrow {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #6b7280; margin-bottom: 0.3rem;
    }
    .section-title { font-size: 1.6rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.01em; margin-bottom: 0.2rem; }
    .section-sub { font-size: 0.9rem; color: #6b7280; margin-bottom: 1.5rem; }
    .q-card {
        border-left: 3px solid #7c3aed; background: rgba(124,58,237,0.05);
        border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .q-type-badge {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        margin-bottom: 6px; display: inline-block;
    }
    .badge-technical  { background:rgba(59,130,246,0.15);  color:#93c5fd; }
    .badge-behavioral { background:rgba(16,185,129,0.15); color:#6ee7b7; }
    .badge-gap        { background:rgba(239,68,68,0.15);   color:#fca5a5; }
    .badge-motivation { background:rgba(245,158,11,0.15);  color:#fcd34d; }
    .feedback-box {
        background: rgba(6,182,212,0.06); border: 1px solid rgba(6,182,212,0.2);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-top: 0.8rem;
    }
    .feedback-row { margin-top: 0.6rem; font-size: 0.92rem; color: #d1d5db; }
    .feedback-row strong { color: #f9fafb; }
    .voice-box {
        background: rgba(139,92,246,0.06); border: 1px solid rgba(139,92,246,0.25);
        border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .jcard {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.1rem 1.2rem 1rem; height: 100%;
        box-sizing: border-box; transition: border-color 0.2s;
    }
    .jcard:hover { border-color: rgba(139,92,246,0.35); }
    .jcard-source {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; padding: 2px 10px; border-radius: 999px;
        background: rgba(16,185,129,0.12); color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.25); display: inline-block; margin-bottom: 8px;
    }
    .jcard-title { font-weight: 700; font-size: 1rem; color: #f0ecff; margin: 4px 0 8px 0; line-height: 1.35; }
    .jcard-meta { font-size: 0.82rem; color: #6b7280; margin: 4px 0; }
    .jcard-btn {
        display: block; margin-top: 14px; text-align: center;
        background: rgba(124,58,237,0.8); color: #fff !important; font-weight: 700;
        font-size: 0.85rem; text-decoration: none !important; border-radius: 10px;
        padding: 9px 0; transition: background 0.2s;
    }
    .jcard-btn:hover { background: #7c3aed; }
    .audit-card {
        background: rgba(255,255,255,0.025); border-radius: 16px;
        padding: 1.2rem 1.4rem; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.07);
    }
    .audit-card.weak   { border-left: 4px solid #ef4444; }
    .audit-card.avg    { border-left: 4px solid #f59e0b; }
    .audit-card.strong { border-left: 4px solid #22c55e; }
    .audit-badge { display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
    .audit-weak   { background:rgba(239,68,68,0.12);   color:#fca5a5; border:1px solid rgba(239,68,68,0.3);   }
    .audit-avg    { background:rgba(245,158,11,0.12);  color:#fcd34d; border:1px solid rgba(245,158,11,0.3);  }
    .audit-strong { background:rgba(34,197,94,0.12);   color:#86efac; border:1px solid rgba(34,197,94,0.3);   }
    .score-bar-bg { background:rgba(255,255,255,0.06); border-radius:999px; height:5px; margin:8px 0 4px 0; }
    .score-bar { height:5px; border-radius:999px; }
    .rewrite-box {
        background: rgba(124,58,237,0.06); border: 1px solid rgba(139,92,246,0.25);
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-top: 0.8rem;
    }
    .improvement-item {
        background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e;
        border-radius: 0 8px 8px 0; padding: 6px 12px; margin: 5px 0;
        font-size: 0.87rem; color: #d1fae5;
    }
    .kw-chip {
        display: inline-block; background: rgba(59,130,246,0.1); color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.25); border-radius: 999px;
        padding: 3px 11px; font-size: 0.75rem; font-weight: 600; margin: 3px;
    }
    .skill-match { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); color:#6ee7b7; border-radius:8px; padding:6px 12px; margin:3px; display:inline-block; font-size:0.82rem; font-weight:600; }
    .skill-miss  { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.3);  color:#fca5a5; border-radius:8px; padding:6px 12px; margin:3px; display:inline-block; font-size:0.82rem; font-weight:600; }
    .skill-extra { background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); color:#93c5fd; border-radius:8px; padding:6px 12px; margin:3px; display:inline-block; font-size:0.82rem; font-weight:600; }
    div[data-testid="stButton"] > button { border-radius: 10px !important; }
    h1, h2, h3 { color: #f9fafb !important; }
    p, li { color: #d1d5db; }
    hr { border-color: rgba(255,255,255,0.07) !important; }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.04) !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #f9fafb !important; border-radius: 10px !important;
    }
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(139,92,246,0.3) !important; border-radius: 14px !important;
    }
    .stInfo    { background: rgba(59,130,246,0.08)  !important; border-color: rgba(59,130,246,0.2)  !important; }
    .stSuccess { background: rgba(16,185,129,0.08)  !important; border-color: rgba(16,185,129,0.2)  !important; }
    .stWarning { background: rgba(245,158,11,0.08)  !important; border-color: rgba(245,158,11,0.2)  !important; }
    .stError   { background: rgba(239,68,68,0.08)   !important; border-color: rgba(239,68,68,0.2)   !important; }
    [data-testid="stChatMessageContent"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "resume_text":     "",
        "jd_text":         "",
        "resume_context":  "",
        "jd_context":      "",
        "analysis":        None,
        "questions":       [],
        "answers":         {},
        "feedbacks":       {},
        "chat_history":    [],
        "step":            "upload",
        "job_title":       "",
        "job_search_meta": None,
        "job_results":     None,
        "audit_result":    None,
        "rewrites":        {},
        "job_query_val":   "",
        "job_loc_val":     "",
        "nlp_result":      None,   # NEW — spaCy analysis result
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_rag_cache: dict = {}

def _get_or_build_rag() -> RAGEngine:
    key = (st.session_state.resume_text, st.session_state.jd_text)
    if "engine" not in _rag_cache or _rag_cache.get("key") != key:
        rag = RAGEngine()
        rag.build_index(st.session_state.resume_text, st.session_state.jd_text)
        _rag_cache["engine"] = rag
        _rag_cache["key"] = key
    return _rag_cache["engine"]

init_state()

# ── Steps — now 7 steps with Skills Map added ─────────────────────────────────

STEPS = {
    "upload":    "Upload",
    "skills":    "Skills Map",   # NEW
    "analyze":   "Analysis",
    "audit":     "Audit",
    "jobs":      "Jobs",
    "questions": "Interview",
    "chat":      "Chat",
}
STEP_KEYS = list(STEPS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_stepper():
    current_idx = STEP_KEYS.index(st.session_state.step)
    nodes = []
    for i, key in enumerate(STEP_KEYS):
        state = "done" if i < current_idx else ("active" if i == current_idx else "")
        icon  = "✓" if i < current_idx else str(i + 1)
        nodes.append(f"""
        <div class="step-node {state}">
            <div class="step-circle">{icon}</div>
            <div class="step-label">{STEPS[key]}</div>
        </div>""")
    st.markdown(f'<div class="stepper-wrap">{"".join(nodes)}</div>', unsafe_allow_html=True)


def color_for_score(score):
    if score >= 75: return "#22c55e"
    if score >= 50: return "#f59e0b"
    return "#ef4444"


def verdict_color(verdict):
    return {
        "Poor Match":   "#ef4444",
        "Fair Match":   "#f97316",
        "Good Match":   "#22c55e",
        "Strong Match": "#06b6d4",
    }.get(verdict, "#9ca3af")


def render_job_card(job: dict):
    source   = job.get("source", "")
    title    = job.get("title", "No Title")
    company  = job.get("company", "Unknown")
    location = job.get("location") or "Not specified"
    salary   = job.get("salary") or "Not specified"
    posted   = job.get("posted") or "N/A"
    url      = job.get("url", "#")
    st.markdown(f"""
<div class="jcard">
  <span class="jcard-source">{source}</span>
  <div class="jcard-title">{title}</div>
  <div class="jcard-meta">🏢 &nbsp;{company}</div>
  <div class="jcard-meta">📍 &nbsp;{location}</div>
  <div class="jcard-meta">💰 &nbsp;{salary}</div>
  <div class="jcard-meta" style="color:#374151">🗓️ &nbsp;{posted}</div>
  <a href="{url}" target="_blank" rel="noopener" class="jcard-btn">Apply Now →</a>
</div>""", unsafe_allow_html=True)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using local Whisper model.
    Saves audio to a temp file, runs Whisper, returns text.
    Completely free — runs on your machine, no API call.
    """
    import whisper
    import soundfile as sf
    import numpy as np

    model = whisper.load_model("base")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:800;color:#f9fafb;margin-bottom:2px'>🎯 CareerLens AI v2</div>"
        "<div style='font-size:0.75rem;color:#4b5563;margin-bottom:1rem'>RAG + NLP + MCP + Voice</div>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("<div style='font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#374151;margin-bottom:0.6rem'>Navigation</div>", unsafe_allow_html=True)

    for step_key, label in STEPS.items():
        is_active = st.session_state.step == step_key
        is_done   = STEP_KEYS.index(step_key) < STEP_KEYS.index(st.session_state.step)
        if is_done or is_active:
            prefix = "✅" if is_done else "▶"
            if st.button(f"{prefix}  {label}", key=f"nav_{step_key}", use_container_width=True):
                st.session_state.step = step_key
                st.rerun()
        else:
            st.markdown(
                f"<div style='padding:6px 12px;font-size:0.9rem;color:#374151;font-weight:500'>○ &nbsp;{label}</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("<div style='font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#374151;margin-bottom:0.6rem'>Tech Stack</div>", unsafe_allow_html=True)
    for item in [
        "🧠 Groq Llama-3.3-70B",
        "📚 LlamaIndex + ChromaDB",
        "🔍 BM25 Hybrid Search",
        "🔤 BGE-small Embeddings",
        "🧬 spaCy NLP (NER)",
        "🎙️ Whisper (Voice STT)",
        "🔌 FastMCP Server",
        "🔎 Adzuna + Remotive",
    ]:
        st.markdown(f"<div style='font-size:0.8rem;color:#4b5563;padding:2px 0'>{item}</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🔄  Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge"><span class="hero-dot"></span>AI Career Copilot v2</div>
    <div class="main-title">CareerLens AI</div>
    <div class="hero-sub">Hybrid RAG · NLP Skill Extraction · Voice Interview · MCP Server · Live Jobs</div>
</div>
""", unsafe_allow_html=True)

render_stepper()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == "upload":

    st.markdown("""
<div class="section-eyebrow">Step 1 of 7</div>
<div class="section-title">Upload your documents</div>
<div class="section-sub">Processed locally — nothing stored externally.</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#9ca3af;margin-bottom:0.5rem'>📄 Resume (PDF)</div>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Upload resume PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_file:
            resume_text = extract_text_from_pdf(pdf_file.read())
            if len(resume_text) < 100:
                st.error("⚠️ Text too short — PDF may be a scanned image.")
            else:
                st.session_state.resume_text = resume_text
                st.markdown(f"<div class='tag tag-green'>✓ {len(resume_text):,} characters extracted</div>", unsafe_allow_html=True)
                with st.expander("Preview extracted text"):
                    st.code(resume_text[:600] + "...", language=None)

    with col2:
        st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#9ca3af;margin-bottom:0.5rem'>💼 Job Description</div>", unsafe_allow_html=True)
        jd_text = st.text_area("Paste job description", height=230,
            placeholder="Paste from LinkedIn, Naukri, or any job board...",
            label_visibility="collapsed")
        if jd_text.strip():
            st.session_state.jd_text = jd_text.strip()
            st.markdown(f"<div class='tag tag-green'>✓ {len(jd_text.strip()):,} characters</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#9ca3af;margin-bottom:0.5rem'>🎯 Target Role</div>", unsafe_allow_html=True)
    job_title = st.text_input("Job title", placeholder="e.g. SDE-1 · Backend Engineer · ML Engineer", label_visibility="collapsed")

    st.divider()

    can_proceed = bool(st.session_state.resume_text and st.session_state.jd_text and job_title.strip())

    col_btn, col_hint = st.columns([1, 2])
    with col_btn:
        go = st.button("Build Index & Continue →", disabled=not can_proceed, type="primary", use_container_width=True)
    with col_hint:
        if not can_proceed:
            missing = []
            if not st.session_state.resume_text: missing.append("resume PDF")
            if not st.session_state.jd_text:     missing.append("job description")
            if not job_title.strip():             missing.append("job title")
            st.markdown(f"<div style='padding-top:8px;font-size:0.85rem;color:#374151'>Still needed: {', '.join(missing)}</div>", unsafe_allow_html=True)

    if go:
        st.session_state.job_title = job_title.strip()
        try:
            with st.spinner("📦 Building ChromaDB hybrid index..."):
                rag = _get_or_build_rag()
                st.session_state.resume_context = rag.get_resume_chunks()
                st.session_state.jd_context     = rag.get_jd_chunks()

            with st.spinner("🧬 Running spaCy NLP skill extraction..."):
                st.session_state.nlp_result = analyze_skills(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                )

            st.session_state.job_query_val = st.session_state.job_title
            st.session_state.job_loc_val   = ""
            st.session_state.step = "skills"
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SKILLS MAP (NEW — spaCy NLP)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "skills":

    nlp = st.session_state.nlp_result or {}

    st.markdown(f"""
<div class="section-eyebrow">Step 2 of 7 · NLP Analysis · {st.session_state.job_title}</div>
<div class="section-title">Skills Map</div>
<div class="section-sub">Extracted by spaCy locally — no LLM call, instant results.</div>
""", unsafe_allow_html=True)

    # ── Top metrics ──
    overlap  = nlp.get("overlap_pct", 0)
    yoe      = nlp.get("years_experience", 0)
    edu      = nlp.get("education", "Not specified")
    seniority = nlp.get("seniority", "Fresher")
    ol_color = color_for_score(overlap)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Skill Overlap</div>
            <div class="metric-value" style="color:{ol_color}">{overlap}%</div>
            <div class="metric-sub">of JD skills covered</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Experience</div>
            <div class="metric-value" style="color:#818cf8">{yoe}</div>
            <div class="metric-sub">years detected</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Education</div>
            <div style="font-size:1.3rem;font-weight:700;color:#f9fafb;margin:0.4rem 0">{edu}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Seniority</div>
            <div style="font-size:1.3rem;font-weight:700;color:#f9fafb;margin:0.4rem 0">{seniority}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Skill breakdown by category ──
    breakdown = nlp.get("skill_breakdown", {})

    if breakdown:
        for category, data in breakdown.items():
            matched = data.get("matched", [])
            missing = data.get("missing", [])
            if not matched and not missing:
                continue

            st.markdown(f"<div style='font-size:0.8rem;font-weight:700;color:#6b7280;margin:1rem 0 0.4rem;text-transform:uppercase;letter-spacing:0.08em'>{category}</div>", unsafe_allow_html=True)

            chips = ""
            for s in matched:
                chips += f'<span class="skill-match">✓ {s}</span>'
            for s in missing:
                chips += f'<span class="skill-miss">✗ {s}</span>'
            st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Extra skills (bonus) ──
    extra = nlp.get("extra_skills", [])
    if extra:
        st.markdown("<div style='font-size:0.8rem;font-weight:700;color:#6b7280;margin:1rem 0 0.4rem;text-transform:uppercase;letter-spacing:0.08em'>Bonus skills (in resume, not required by JD)</div>", unsafe_allow_html=True)
        chips = " ".join([f'<span class="skill-extra">{s}</span>' for s in extra])
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("""
<div style='margin-top:1.5rem;padding:1rem;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;font-size:0.85rem;color:#6b7280'>
    <strong style='color:#9ca3af'>🧬 How this works:</strong>
    spaCy scans your resume and JD text using a 500+ skill vocabulary (PhraseMatcher).
    Green = skill found in both. Red = JD requires it but your resume doesn't mention it.
    This runs locally in milliseconds — no API call, no token cost.
</div>
""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.step = "upload"; st.rerun()
    with col2:
        if st.button("Run LLM Analysis →", use_container_width=True, type="primary"):
            with st.spinner("🤖 Analyzing resume against job description..."):
                analysis = analyze_resume_vs_jd(
                    st.session_state.resume_context,
                    st.session_state.jd_context,
                    st.session_state.job_title,
                )
                st.session_state.analysis = analysis
            st.session_state.step = "analyze"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ANALYSIS (LLM)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "analyze":
    a = st.session_state.analysis

    st.markdown(f"""
<div class="section-eyebrow">Step 3 of 7 · LLM Analysis · {st.session_state.job_title}</div>
<div class="section-title">How well does your resume match?</div>
""", unsafe_allow_html=True)

    score   = a.get("match_score", 0)
    verdict = a.get("match_verdict", "")
    vc      = verdict_color(verdict)
    sc      = color_for_score(score)

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Match Score</div>
            <div class="metric-value" style="color:{sc}">{score}</div>
            <div class="metric-sub">out of 100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Verdict</div>
            <div style="font-size:1.4rem;font-weight:700;color:{vc};margin:0.3rem 0">{verdict}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="card">
            <div class="metric-label">Summary</div>
            <div style="font-size:0.92rem;color:#d1d5db;margin-top:4px;line-height:1.6">{a.get("one_line_summary","")}</div>
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#9ca3af;margin-bottom:0.5rem'>✅ Strengths</div>", unsafe_allow_html=True)
        for s in a.get("top_strengths", []):
            st.markdown(f"<div style='font-size:0.9rem;color:#d1d5db;padding:3px 0'>• {s}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#9ca3af;margin-bottom:0.5rem'>🔑 Missing Keywords</div>", unsafe_allow_html=True)
        kws = a.get("missing_keywords", [])
        st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in kws]), unsafe_allow_html=True)

    with col_r:
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#9ca3af;margin-bottom:0.5rem'>⚠️ Critical Gaps</div>", unsafe_allow_html=True)
        for g in a.get("critical_gaps", []):
            st.markdown(f"<div style='font-size:0.9rem;color:#d1d5db;padding:3px 0'>• {g}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#9ca3af;margin-bottom:0.5rem'>📝 Quick Improvements</div>", unsafe_allow_html=True)
        for tip in a.get("resume_improvements", []):
            st.markdown(f"<div style='font-size:0.9rem;color:#d1d5db;padding:3px 0'>• {tip}</div>", unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🩺 Audit Resume", use_container_width=True, type="primary"):
            st.session_state.step = "audit"; st.rerun()
    with c2:
        if st.button("🔎 Find Jobs", use_container_width=True):
            st.session_state.step = "jobs"; st.rerun()
    with c3:
        if st.button("🎤 Interview Prep", use_container_width=True):
            if not st.session_state.questions:
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_interview_questions(
                        st.session_state.resume_context, st.session_state.jd_context, st.session_state.job_title)
            st.session_state.step = "questions"; st.rerun()
    with c4:
        if st.button("💬 Career Chat", use_container_width=True):
            st.session_state.step = "chat"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RESUME AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "audit":

    st.markdown(f"""
<div class="section-eyebrow">Step 4 of 7 · Resume Audit · {st.session_state.job_title}</div>
<div class="section-title">Section-by-section health check</div>
<div class="section-sub">AI scans every section, flags weak spots, and can auto-rewrite them.</div>
""", unsafe_allow_html=True)

    if st.session_state.audit_result is None:
        with st.spinner("🔬 Scanning resume section by section..."):
            st.session_state.audit_result = audit_resume_sections(
                st.session_state.resume_text, st.session_state.job_title)
        st.rerun()

    audit    = st.session_state.audit_result
    sections = audit.get("sections", [])

    overall_score  = audit.get("overall_score", 0)
    overall_health = audit.get("overall_health", "Unknown")
    ats_score      = audit.get("ats_score", 0)
    ats_issues     = audit.get("ats_issues", [])
    priority_fix   = audit.get("top_priority_fix", "")

    health_color = {"Needs Work": "#ef4444", "Decent": "#f59e0b", "Strong": "#22c55e"}.get(overall_health, "#9ca3af")
    ats_color    = color_for_score(ats_score)

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Resume Health</div>
            <div class="metric-value" style="color:{health_color}">{overall_score}</div>
            <div class="metric-sub" style="color:{health_color}">{overall_health}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ATS Score</div>
            <div class="metric-value" style="color:{ats_color}">{ats_score}</div>
            <div class="metric-sub">Applicant Tracking System</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        issues_html = "".join([f"<div style='color:#fca5a5;font-size:0.82rem;margin:3px 0'>⚠ {i}</div>" for i in ats_issues[:3]])
        st.markdown(f"""<div class="card">
            <div class="metric-label">ATS Issues</div>
            {issues_html if issues_html else '<div style="color:#6ee7b7;font-size:0.85rem">✓ No major issues</div>'}
        </div>""", unsafe_allow_html=True)

    if priority_fix:
        st.warning(f"**Top priority fix:** {priority_fix}")

    weak_c   = sum(1 for s in sections if s.get("rating") == "Weak")
    avg_c    = sum(1 for s in sections if s.get("rating") == "Average")
    strong_c = sum(1 for s in sections if s.get("rating") == "Strong")
    cm1, cm2, cm3 = st.columns(3)
    with cm1: st.metric("🔴 Weak", weak_c)
    with cm2: st.metric("🟡 Average", avg_c)
    with cm3: st.metric("🟢 Strong", strong_c)

    st.markdown("---")

    CARD_CLASS  = {"Weak": "weak",     "Average": "avg",      "Strong": "strong"}
    BADGE_CLASS = {"Weak": "audit-weak","Average": "audit-avg","Strong": "audit-strong"}
    BAR_COLOR   = {"Weak": "#ef4444",  "Average": "#f59e0b",  "Strong": "#22c55e"}
    EMOJI       = {"Weak": "🔴", "Average": "🟡", "Strong": "🟢"}

    for sec in sections:
        name    = sec.get("name", "Unknown")
        rating  = sec.get("rating", "Average")
        score   = sec.get("score", 50)
        issues  = sec.get("issues", [])
        wins    = sec.get("quick_wins", [])
        content = sec.get("current_content", "")
        emoji   = EMOJI.get(rating, "🟡")

        with st.expander(f"{emoji} **{name}** — {rating} ({score}/100)", expanded=(rating == "Weak")):
            issues_html = "".join([f"<div style='color:#fca5a5;font-size:0.83rem;margin:3px 0 3px 10px'>• {i}</div>" for i in issues])
            wins_html   = "".join([f"<div style='color:#6ee7b7;font-size:0.83rem;margin:3px 0 3px 10px'>• {w}</div>" for w in wins])
            st.markdown(f"""
<div class="audit-card {CARD_CLASS.get(rating,'avg')}">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px">
    <span style="font-size:1.05rem;font-weight:700;color:#f9fafb">{name}</span>
    <span class="audit-badge {BADGE_CLASS.get(rating,'audit-avg')}">{emoji} {rating}</span>
  </div>
  <div class="score-bar-bg"><div class="score-bar" style="width:{score}%;background:{BAR_COLOR.get(rating,'#f59e0b')}"></div></div>
  <div style="font-size:0.75rem;color:#4b5563;margin-bottom:10px">Score: {score}/100</div>
  {'<div style="color:#fca5a5;font-size:0.8rem;font-weight:600;margin-bottom:3px">Issues</div>' + issues_html if issues else ''}
  {'<div style="color:#6ee7b7;font-size:0.8rem;font-weight:600;margin-top:10px;margin-bottom:3px">Quick wins</div>' + wins_html if wins else ''}
</div>""", unsafe_allow_html=True)

            if content:
                with st.expander("📄 View current content"):
                    st.text_area("", value=content, height=110, disabled=True, key=f"content_{name}", label_visibility="collapsed")

            rw_key = name
            btn1, btn2, btn3 = st.columns([2, 2, 3])
            with btn1:
                do_rw = st.button("✨ Auto-Rewrite", key=f"rw_{name}", type="primary", use_container_width=True)
            with btn2:
                do_sg = st.button("💡 Suggestions Only", key=f"sg_{name}", use_container_width=True)
            with btn3:
                if rw_key in st.session_state.rewrites:
                    if st.button("🗑 Clear", key=f"cl_{name}", use_container_width=True):
                        del st.session_state.rewrites[rw_key]; st.rerun()

            if do_rw or do_sg:
                with st.spinner(f"{'Rewriting' if do_rw else 'Generating suggestions for'} {name}..."):
                    result = rewrite_resume_section(
                        section_name=name, current_content=content,
                        resume_text=st.session_state.resume_text,
                        job_title=st.session_state.job_title, issues=issues,
                    )
                    result["_mode"] = "rewrite" if do_rw else "suggest"
                    st.session_state.rewrites[rw_key] = result
                st.rerun()

            if rw_key in st.session_state.rewrites:
                rw   = st.session_state.rewrites[rw_key]
                mode = rw.get("_mode", "rewrite")
                st.markdown('<div class="rewrite-box">', unsafe_allow_html=True)
                if mode == "rewrite" and rw.get("rewritten"):
                    st.markdown("**✨ AI-Rewritten Version**")
                    st.info("Copy this and replace your current section.")
                    st.code(rw["rewritten"], language=None)
                bullets = rw.get("bullet_suggestions", [])
                if bullets:
                    st.markdown("**💡 Alternative Bullet Points**")
                    for i, b in enumerate(bullets, 1):
                        st.markdown(f"<div style='background:rgba(124,58,237,0.07);border-left:3px solid #7c3aed;border-radius:0 8px 8px 0;padding:8px 14px;margin:5px 0;font-size:0.9rem;color:#e9e4ff'>{i}. {b}</div>", unsafe_allow_html=True)
                improvements = rw.get("improvements_made", [])
                if improvements:
                    st.markdown("**📈 What Was Improved**")
                    for imp in improvements:
                        st.markdown(f'<div class="improvement-item">✅ {imp}</div>', unsafe_allow_html=True)
                keywords = rw.get("keywords_added", [])
                if keywords:
                    st.markdown("**🔑 Keywords Added**")
                    st.markdown(" ".join([f'<span class="kw-chip">{k}</span>' for k in keywords]), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("← Analysis", use_container_width=True):
            st.session_state.step = "analyze"; st.rerun()
    with n2:
        if st.button("🔎 Find Jobs", use_container_width=True):
            st.session_state.step = "jobs"; st.rerun()
    with n3:
        if st.button("🎤 Interview Prep →", use_container_width=True, type="primary"):
            if not st.session_state.questions:
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_interview_questions(
                        st.session_state.resume_context, st.session_state.jd_context, st.session_state.job_title)
            st.session_state.step = "questions"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — LIVE JOBS
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "jobs":

    st.markdown(f"""
<div class="section-eyebrow">Step 5 of 7 · Live Jobs · {st.session_state.job_title}</div>
<div class="section-title">Real, current openings for you</div>
<div class="section-sub">Click Apply Now to go directly to the listing.</div>
""", unsafe_allow_html=True)

    if st.session_state.job_search_meta is None:
        with st.spinner("🧠 Building smart search query from your resume..."):
            st.session_state.job_search_meta = extract_job_search_query(
                st.session_state.resume_context, st.session_state.job_title)
        meta = st.session_state.job_search_meta or {}
        st.session_state.job_query_val = meta.get("search_query", st.session_state.job_title)
        st.session_state.job_loc_val   = meta.get("likely_location", "")

    meta = st.session_state.job_search_meta or {}
    if meta.get("top_skills"):
        st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in meta["top_skills"]]), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        query = st.text_input("Search query", value=st.session_state.job_query_val, key="job_q")
    with c2:
        location = st.text_input("Location (optional)", value=st.session_state.job_loc_val, key="job_loc")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        remote_only = st.checkbox("Remote only", value=False)

    if st.button("🔄 Search Live Jobs", type="primary", use_container_width=True):
        st.session_state.job_query_val = query
        st.session_state.job_loc_val   = location
        with st.spinner("🔎 Fetching live listings..."):
            st.session_state.job_results = get_job_matches(query, location, remote_only)

    if st.session_state.job_results is None:
        with st.spinner("🔎 Fetching live listings..."):
            st.session_state.job_results = get_job_matches(
                st.session_state.job_query_val, st.session_state.job_loc_val, False)

    results = st.session_state.job_results or {"jobs": [], "sources_used": [], "adzuna_configured": False}
    jobs    = results.get("jobs", [])

    if not results.get("adzuna_configured"):
        st.info("💡 Showing remote jobs (Remotive). For India + global, add a free Adzuna key.")

    if not jobs:
        st.warning("No listings found — try a broader query.")
    else:
        sources_str = " & ".join(results.get("sources_used", []))
        st.markdown(f"<div class='tag tag-green'>✓ {len(jobs)} openings via {sources_str}</div><div style='height:0.8rem'></div>", unsafe_allow_html=True)
        for row_start in range(0, len(jobs), 3):
            row_jobs  = jobs[row_start: row_start + 3]
            grid_cols = st.columns(3)
            for col, job in zip(grid_cols, row_jobs):
                with col:
                    render_job_card(job)

    st.divider()
    n1, n2 = st.columns(2)
    with n1:
        if st.button("← Analysis", use_container_width=True):
            st.session_state.step = "analyze"; st.rerun()
    with n2:
        if st.button("🎤 Interview Prep →", use_container_width=True, type="primary"):
            if not st.session_state.questions:
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_interview_questions(
                        st.session_state.resume_context, st.session_state.jd_context, st.session_state.job_title)
            st.session_state.step = "questions"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — INTERVIEW PREP (with Voice — Whisper)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "questions":

    st.markdown(f"""
<div class="section-eyebrow">Step 6 of 7 · Interview Prep · {st.session_state.job_title}</div>
<div class="section-title">10 questions tailored to your resume</div>
<div class="section-sub">Type your answer OR record your voice — Whisper transcribes it locally.</div>
""", unsafe_allow_html=True)

    questions = st.session_state.questions

    BADGE = {
        "Technical":  "badge-technical",
        "Behavioral": "badge-behavioral",
        "Gap-based":  "badge-gap",
        "Motivation": "badge-motivation",
    }

    # Try importing mic recorder — graceful fallback if not installed
    try:
        from streamlit_mic_recorder import mic_recorder
        voice_available = True
    except ImportError:
        voice_available = False

    for q in questions:
        qid      = q.get("id", 0)
        qtype    = q.get("type", "")
        question = q.get("question", "")
        why      = q.get("why_asked", "")
        hint     = q.get("hint", "")
        bclass   = BADGE.get(qtype, "badge-technical")

        with st.expander(f"Q{qid}. {question[:85]}{'...' if len(question) > 85 else ''}", expanded=(qid == 1)):

            st.markdown(f"""<div class="q-card">
                <span class="q-type-badge {bclass}">{qtype}</span>
                <div style="font-size:1.02rem;font-weight:600;color:#f9fafb;margin:6px 0 8px 0;line-height:1.5">{question}</div>
                <div style="font-size:0.83rem;color:#6b7280;font-style:italic">💡 {why}</div>
            </div>""", unsafe_allow_html=True)

            with st.expander("Show hint"):
                st.info(hint)

            # ── Answer input: text OR voice ──
            tab_text, tab_voice = st.tabs(["⌨️ Type answer", "🎙️ Voice answer (Whisper)"])

            with tab_text:
                answer = st.text_area(
                    "Your answer:", key=f"ans_{qid}", height=140,
                    placeholder="Type your answer as you would say it in a real interview...",
                )
                st.session_state.answers[qid] = answer

            with tab_voice:
                if voice_available:
                    st.markdown("<div class='voice-box'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-size:0.85rem;color:#9ca3af;margin-bottom:0.5rem'>🎙️ Click Record, speak your answer, click Stop. Whisper will transcribe it locally.</div>", unsafe_allow_html=True)

                    audio = mic_recorder(
                        start_prompt="🔴 Start Recording",
                        stop_prompt="⬛ Stop Recording",
                        key=f"mic_{qid}",
                    )

                    if audio and audio.get("bytes"):
                        with st.spinner("🎙️ Whisper is transcribing your answer locally..."):
                            try:
                                transcribed = transcribe_audio(audio["bytes"])
                                st.success(f"**Transcribed:** {transcribed}")
                                st.session_state.answers[qid] = transcribed
                            except Exception as e:
                                st.error(f"Transcription failed: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("Install `streamlit-mic-recorder` to enable voice: `pip install streamlit-mic-recorder`")

            # ── AI Feedback button ──
            final_answer = st.session_state.answers.get(qid, "")
            col_fb, _ = st.columns([1, 3])
            with col_fb:
                if st.button("Get AI Feedback", key=f"fb_{qid}", type="primary"):
                    if not final_answer.strip():
                        st.warning("Type or record your answer first.")
                    else:
                        with st.spinner("Evaluating..."):
                            fb = evaluate_answer(question, final_answer, st.session_state.job_title, st.session_state.resume_context)
                            st.session_state.feedbacks[qid] = fb

            if qid in st.session_state.feedbacks:
                fb    = st.session_state.feedbacks[qid]
                score = fb.get("score", 0)
                label = fb.get("score_label", "")
                sc    = {"Weak":"#ef4444","Needs Work":"#f97316","Decent":"#eab308","Good":"#22c55e","Excellent":"#06b6d4"}.get(label, "#9ca3af")
                st.markdown(f"""<div class="feedback-box">
                    <div style="font-size:1.1rem;font-weight:700;color:{sc}">Score: {score}/10 — {label}</div>
                    <div class="feedback-row"><strong>👍 What was good:</strong><br>{fb.get("what_was_good","")}</div>
                    <div class="feedback-row"><strong>⚠️ What was missing:</strong><br>{fb.get("what_was_missing","")}</div>
                    <div class="feedback-row"><strong>📋 Model answer outline:</strong><br>{fb.get("model_answer_outline","")}</div>
                    <div class="feedback-row" style="color:#a5b4fc"><strong>⭐ STAR tip:</strong> {fb.get("star_tip","")}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("← Analysis", use_container_width=True):
            st.session_state.step = "analyze"; st.rerun()
    with n2:
        if st.button("🔎 Job Matches", use_container_width=True):
            st.session_state.step = "jobs"; st.rerun()
    with n3:
        if st.button("💬 Career Chat →", use_container_width=True, type="primary"):
            st.session_state.step = "chat"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == "chat":

    st.markdown(f"""
<div class="section-eyebrow">Step 7 of 7 · Career Chat · {st.session_state.job_title}</div>
<div class="section-title">Ask me anything about your resume</div>
<div class="section-sub">Questions, salary advice, how to handle tough interview moments.</div>
""", unsafe_allow_html=True)

    SUGGESTIONS = [
        "What should I highlight in my intro?",
        "What skills should I learn next?",
        "How do I explain project gaps?",
        "What salary range should I expect?",
        "Which projects to talk about first?",
        "How to answer 'Why this company'?",
    ]

    st.markdown("<div style='font-size:0.8rem;font-weight:700;color:#4b5563;margin-bottom:0.5rem;letter-spacing:0.05em;text-transform:uppercase'>Quick questions</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                with st.spinner("Thinking..."):
                    reply = chat_with_resume(sug, st.session_state.resume_context,
                        st.session_state.jd_context, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask CareerLens anything about your resume or this role...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            reply = chat_with_resume(user_input, st.session_state.resume_context,
                st.session_state.jd_context, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    st.divider()
    n1, n2 = st.columns(2)
    with n1:
        if st.button("← Interview Questions", use_container_width=True):
            st.session_state.step = "questions"; st.rerun()
    with n2:
        if st.button("🔎 Job Matches", use_container_width=True):
            st.session_state.step = "jobs"; st.rerun()
