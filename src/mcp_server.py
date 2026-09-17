"""
MCP Server — CareerLens AI
Exposes CareerLens functions as MCP (Model Context Protocol) tools.

What is MCP?
  MCP is an open standard by Anthropic that lets AI assistants (Claude,
  Cursor, any MCP client) call external tools as plugins.

  Without MCP: CareerLens only works as a Streamlit web app.
  With MCP:    Any AI assistant can call CareerLens programmatically.

How to run this server:
  python src/mcp_server.py

How to connect (Claude Desktop example):
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "careerlens": {
        "command": "python",
        "args": ["C:/path/to/careerlens/src/mcp_server.py"]
      }
    }
  }

Tools exposed:
  1. analyze_resume    — match score, strengths, gaps, missing keywords
  2. audit_resume      — section-by-section health check
  3. generate_questions — 10 personalized interview questions
  4. evaluate_answer   — score a spoken/written interview answer
  5. get_jobs          — live job listings from Adzuna + Remotive
  6. extract_skills    — NLP skill extraction without calling the LLM

Why FastMCP?
  FastMCP is the simplest way to build MCP servers in Python.
  One decorator (@mcp.tool) turns any function into an MCP tool.
  FastMCP handles all the protocol, serialization, and transport.
"""

import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from src.llm_agent import (
    analyze_resume_vs_jd,
    generate_interview_questions,
    evaluate_answer,
    audit_resume_sections,
)
from src.job_search import get_job_matches
from src.nlp_engine import analyze_skills

# ── Create MCP server ─────────────────────────────────────────────────────────
# FastMCP takes a name — this is what appears in the AI assistant's tool list

mcp = FastMCP("CareerLens AI")


# ── Tool 1: Analyze Resume ────────────────────────────────────────────────────

@mcp.tool()
def analyze_resume(
    resume_text: str,
    job_description: str,
    job_title: str,
) -> dict:
    """
    Analyze a resume against a job description.

    Returns match score (0-100), verdict, top strengths,
    critical gaps, missing keywords, and improvement tips.

    Args:
        resume_text:     Full text of the candidate's resume
        job_description: Full text of the job description
        job_title:       Target role (e.g. "Backend Engineer")
    """
    # For MCP we pass text directly — no RAG chunking needed
    # The LLM gets the full text since MCP callers handle context themselves
    result = analyze_resume_vs_jd(
        resume_context=resume_text[:3000],
        jd_context=job_description[:2000],
        job_title=job_title,
    )
    return result


# ── Tool 2: Audit Resume ──────────────────────────────────────────────────────

@mcp.tool()
def audit_resume(
    resume_text: str,
    job_title: str,
) -> dict:
    """
    Section-by-section resume health check.

    Scans Summary, Work Experience, Skills, Projects, Education.
    Rates each section as Strong / Average / Weak.
    Returns ATS score and top priority fix.

    Args:
        resume_text: Full text of the candidate's resume
        job_title:   Target role for context
    """
    return audit_resume_sections(
        resume_text=resume_text,
        job_title=job_title,
    )


# ── Tool 3: Generate Interview Questions ──────────────────────────────────────

@mcp.tool()
def generate_questions(
    resume_text: str,
    job_description: str,
    job_title: str,
) -> list:
    """
    Generate 10 personalized interview questions based on the resume and JD.

    Mix of: Technical (4), Behavioral (3), Gap-based (2), Motivation (1).
    Each question includes: why it's asked + what a strong answer covers.

    Args:
        resume_text:     Full text of the candidate's resume
        job_description: Full text of the job description
        job_title:       Target role
    """
    return generate_interview_questions(
        resume_context=resume_text[:3000],
        jd_context=job_description[:2000],
        job_title=job_title,
    )


# ── Tool 4: Evaluate Interview Answer ─────────────────────────────────────────

@mcp.tool()
def evaluate_interview_answer(
    question: str,
    answer: str,
    job_title: str,
    resume_text: str = "",
) -> dict:
    """
    Score a candidate's interview answer out of 10.

    Returns score, label (Weak/Decent/Good/Excellent),
    what was good, what was missing, model answer outline, STAR tip.

    Args:
        question:    The interview question asked
        answer:      The candidate's answer (typed or transcribed from voice)
        job_title:   Target role for context
        resume_text: Optional resume text for personalized feedback
    """
    return evaluate_answer(
        question=question,
        answer=answer,
        job_title=job_title,
        resume_context=resume_text[:800],
    )


# ── Tool 5: Get Live Jobs ─────────────────────────────────────────────────────

@mcp.tool()
def get_live_jobs(
    search_query: str,
    location: str = "",
    remote_only: bool = False,
) -> dict:
    """
    Fetch live job listings from Adzuna and Remotive job boards.

    Returns list of jobs with: title, company, location, salary, url.
    Adzuna covers India + global. Remotive covers remote tech jobs.

    Args:
        search_query: Job title or keywords (e.g. "Python Backend Engineer")
        location:     City or region (e.g. "Bangalore") — optional
        remote_only:  If True, only return remote jobs from Remotive
    """
    return get_job_matches(
        search_query=search_query,
        location=location,
        remote_only=remote_only,
    )


# ── Tool 6: Extract Skills (NLP, no LLM) ─────────────────────────────────────

@mcp.tool()
def extract_resume_skills(
    resume_text: str,
    job_description: str,
) -> dict:
    """
    Extract and compare skills using local NLP (gazetteer matching) — no LLM call needed.

    Faster and cheaper than the LLM-based analysis for skill gap detection.
    Returns matched skills, missing skills, and overlap percentage.

    Args:
        resume_text:     Full text of the candidate's resume
        job_description: Full text of the job description
    """
    return analyze_skills(
        resume_text=resume_text,
        jd_text=job_description,
    )


# ── Run server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CareerLens MCP Server starting...")
    print("Tools available:")
    print("  - analyze_resume")
    print("  - audit_resume")
    print("  - generate_questions")
    print("  - evaluate_interview_answer")
    print("  - get_live_jobs")
    print("  - extract_resume_skills")
    print()
    print("Connect via Claude Desktop or any MCP client.")
    mcp.run()
