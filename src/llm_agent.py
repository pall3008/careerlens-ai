"""
LLM Agent - All Groq API calls and prompt engineering
Handles: analysis, question generation, answer feedback, job-search query extraction,
         resume section audit + auto-rewrite
"""

import os
import json
import re
from groq import Groq

try:
    from src.config import get_secret, available_secret_names
except ImportError:            # when this file is run directly from inside src/
    from config import get_secret, available_secret_names

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = "openai/gpt-oss-20b"    # available on your Groq account, good JSON output
MAX_TOKENS  = 1024                    # safe under free tier limits
TEMPERATURE = 0.7


# ── Client ────────────────────────────────────────────────────────────────────

def get_client() -> Groq:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        visible = available_secret_names()
        seen = ", ".join(visible) if visible else "none at all"
        raise ValueError(
            "GROQ_API_KEY not found.\n\n"
            f"Secrets this app can currently see: {seen}\n\n"
            "  Running locally: copy .env.example to .env and add your key.\n"
            "  Deployed:        Manage app -> Settings -> Secrets. Paste the key\n"
            "                   at the TOP of the box, above any [section] header,\n"
            "                   as:  GROQ_API_KEY = \"gsk_...\"\n"
            "                   Save, wait ~1 minute, then reboot the app.\n"
            "  Free key at:     https://console.groq.com"
        )
    return Groq(api_key=api_key)


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = TEMPERATURE) -> str:
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(
            f"Groq API call failed: {e}\n\n"
            "If this mentions 'model_decommissioned', check https://console.groq.com/docs/deprecations "
            "for the current model name and update MODEL in src/llm_agent.py."
        ) from e
    return response.choices[0].message.content.strip()


def _parse_json(raw: str, fallback=None):
    match = re.search(r'\{.*\}|\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not parse JSON from LLM response:\n{raw}")


# ── Existing Agent Functions ──────────────────────────────────────────────────

def analyze_resume_vs_jd(resume_context: str, jd_context: str, job_title: str) -> dict:
    system = """You are a senior technical recruiter and career coach with 15 years of experience 
at top product-based companies. You give honest, actionable feedback.
Always respond with valid JSON only — no markdown, no explanation outside the JSON."""

    user = f"""
Analyze this candidate's resume against the job description for: {job_title}

RESUME CONTENT:
{resume_context}

JOB DESCRIPTION:
{jd_context}

Respond with this exact JSON structure:
{{
  "match_score": <integer 0-100>,
  "match_verdict": "<Poor Match | Fair Match | Good Match | Strong Match>",
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "critical_gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "missing_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>", "<keyword 5>"],
  "resume_improvements": ["<tip 1>", "<tip 2>", "<tip 3>"],
  "one_line_summary": "<honest one sentence verdict>"
}}
"""
    raw = _call_llm(system, user, temperature=0.3)
    return _parse_json(raw)


def generate_interview_questions(resume_context: str, jd_context: str, job_title: str) -> list:
    system = """You are a technical interviewer at a top product-based company (think Google, Flipkart, Razorpay).
You generate sharp, specific interview questions — not generic ones.
Always respond with valid JSON only."""

    user = f"""
Generate 10 interview questions for a candidate applying for: {job_title}

CANDIDATE RESUME:
{resume_context}

JOB DESCRIPTION:
{jd_context}

Rules:
- 4 technical questions (based on their actual tech stack from resume)
- 3 behavioral questions (based on their projects/experience)
- 2 gap questions (probe weak areas or missing skills)
- 1 motivation question

Respond with this exact JSON:
{{
  "questions": [
    {{
      "id": 1,
      "type": "<Technical | Behavioral | Gap-based | Motivation>",
      "question": "<the interview question>",
      "why_asked": "<one sentence: what the interviewer is probing>",
      "hint": "<what a strong answer should cover>"
    }}
  ]
}}
"""
    raw = _call_llm(system, user, temperature=0.5)
    data = _parse_json(raw, fallback={"questions": []})
    return data.get("questions", [])


def evaluate_answer(question: str, answer: str, job_title: str, resume_context: str) -> dict:
    system = """You are an experienced technical interviewer giving honest, constructive feedback.
Be encouraging but direct. Always respond with valid JSON only."""

    user = f"""
Evaluate this interview answer for a {job_title} role.

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

CANDIDATE'S BACKGROUND (from resume):
{resume_context[:800]}

Respond with this JSON:
{{
  "score": <integer 1-10>,
  "score_label": "<Weak | Needs Work | Decent | Good | Excellent>",
  "what_was_good": "<specific praise>",
  "what_was_missing": "<specific gap in the answer>",
  "model_answer_outline": "<how an ideal answer would be structured>",
  "star_tip": "<one tip to improve using STAR method if applicable>"
}}
"""
    raw = _call_llm(system, user, temperature=0.4)
    return _parse_json(raw)


def chat_with_resume(query: str, resume_context: str, jd_context: str, chat_history: list) -> str:
    system = """You are CareerLens AI, a career advisor helping a job seeker.
You have access to their resume and the job description they're targeting.
Be specific, use their actual experience, and give actionable advice.
Keep responses concise (under 200 words) unless a detailed breakdown is needed."""

    messages = [{"role": "system", "content": system}]
    context_msg = f"""Here is the candidate's context:

RESUME:
{resume_context[:1500]}

JOB DESCRIPTION:
{jd_context[:1000]}

Help the candidate with their questions."""
    messages.append({"role": "user", "content": context_msg})
    messages.append({"role": "assistant", "content": "Got it! I've reviewed your resume and the job description. What would you like to know?"})

    for msg in chat_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}") from e
    return response.choices[0].message.content.strip()


def extract_job_search_query(resume_context: str, job_title: str) -> dict:
    system = """You convert a candidate's resume into a concise, realistic job search query.
Always respond with valid JSON only — no markdown, no explanation outside the JSON."""

    user = f"""
TARGET ROLE: {job_title}

RESUME CONTENT:
{resume_context}

Respond with this exact JSON structure:
{{
  "search_query": "<2-5 word job title/keyword string ideal for a job board search>",
  "top_skills": ["<skill1>", "<skill2>", "<skill3>", "<skill4>", "<skill5>"],
  "likely_location": "<city/region guessed from resume content, or '' if you cannot tell>",
  "experience_level": "<Fresher | Junior | Mid | Senior>"
}}
"""
    raw = _call_llm(system, user, temperature=0.2)
    return _parse_json(raw, fallback={
        "search_query": job_title,
        "top_skills": [],
        "likely_location": "",
        "experience_level": "Fresher",
    })


# ── NEW: Resume Section Auditor ───────────────────────────────────────────────

def audit_resume_sections(resume_text: str, job_title: str) -> dict:
    """
    Scans the full resume and returns a health report for each major section.
    Rates each section as Strong / Average / Weak and explains why.
    """
    system = """You are an elite resume coach who has reviewed 10,000+ resumes for top tech companies.
You identify weak sections with surgical precision and give concrete, actionable feedback.
Always respond with valid JSON only — no markdown, no preamble, no explanation outside the JSON."""

    user = f"""
Audit this resume for a candidate targeting: {job_title}

FULL RESUME TEXT:
{resume_text}

Analyse these sections: Summary/Objective, Work Experience, Skills, Projects, Education, Achievements/Awards (if present).
For each section found in the resume, evaluate its quality.

Respond with this EXACT JSON:
{{
  "overall_health": "<Needs Work | Decent | Strong>",
  "overall_score": <integer 0-100>,
  "sections": [
    {{
      "name": "<section name, e.g. 'Summary', 'Work Experience', 'Skills', 'Projects', 'Education'>",
      "present": <true | false>,
      "rating": "<Weak | Average | Strong>",
      "score": <integer 0-100>,
      "issues": ["<specific issue 1>", "<specific issue 2>"],
      "quick_wins": ["<quick fix 1>", "<quick fix 2>"],
      "current_content": "<paste the exact text from resume that belongs to this section, max 400 chars>"
    }}
  ],
  "top_priority_fix": "<the single most impactful change to make right now>",
  "ats_score": <integer 0-100>,
  "ats_issues": ["<ATS issue 1>", "<ATS issue 2>"]
}}
"""
    raw = _call_llm(system, user, temperature=0.2)
    return _parse_json(raw, fallback={"sections": [], "overall_health": "Unknown", "overall_score": 0})


def rewrite_resume_section(
    section_name: str,
    current_content: str,
    resume_text: str,
    job_title: str,
    issues: list,
) -> dict:
    """
    Auto-rewrites a weak resume section using the candidate's own information.
    Returns both a rewritten version and bullet-point suggestions.
    """
    system = """You are an expert resume writer who crafts ATS-friendly, impact-driven resume content.
You ONLY use information already present in the candidate's resume — you never fabricate facts, numbers, or experiences.
Write in professional first-person-implied style (no "I"). Use strong action verbs.
Always respond with valid JSON only."""

    issues_str = "\n".join(f"- {i}" for i in issues) if issues else "- Needs to be more impactful and specific"

    user = f"""
Rewrite the "{section_name}" section of this resume for a {job_title} role.

FULL RESUME (for context — use only real info from here):
{resume_text[:2500]}

CURRENT "{section_name}" CONTENT:
{current_content}

IDENTIFIED ISSUES:
{issues_str}

Rules:
1. Use ONLY facts, skills, projects, and experience already in the resume
2. Do NOT invent numbers, companies, or achievements not mentioned
3. Use strong action verbs and quantify wherever the resume provides numbers
4. Make it ATS-friendly with relevant keywords for {job_title}
5. Keep it concise and punchy

Respond with this EXACT JSON:
{{
  "rewritten": "<the full rewritten section text, formatted professionally>",
  "bullet_suggestions": [
    "<alternative bullet point 1 — pick the best one>",
    "<alternative bullet point 2>",
    "<alternative bullet point 3>"
  ],
  "improvements_made": ["<what changed and why 1>", "<what changed and why 2>", "<what changed and why 3>"],
  "keywords_added": ["<keyword 1>", "<keyword 2>", "<keyword 3>"]
}}
"""
    raw = _call_llm(system, user, temperature=0.4)
    return _parse_json(raw, fallback={
        "rewritten": "",
        "bullet_suggestions": [],
        "improvements_made": [],
        "keywords_added": [],
    })
