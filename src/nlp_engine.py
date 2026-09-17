"""
NLP Engine — CareerLens AI
Uses spaCy for fast, local, zero-cost skill extraction from resume + JD text.

Why spaCy instead of asking the LLM?
  - Runs locally — no API call, no token cost
  - Deterministic — same input always gives same output
  - Fast — processes a resume in milliseconds
  - LLMs sometimes miss exact technical keywords; spaCy + PhraseMatcher never does

What this file produces:
  - Skills found in resume
  - Skills required in JD
  - Matched skills (strengths)
  - Missing skills (gaps)
  - Overlap percentage
  - Years of experience
  - Education level
  - Seniority level
"""

import re
import spacy
from typing import List

# ── Load spaCy model ──────────────────────────────────────────────────────────
# en_core_web_sm is a small 12MB English model — downloads once via:
#   python -m spacy download en_core_web_sm
# It handles tokenization, POS tagging, NER (Named Entity Recognition)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ── Skills Taxonomy ───────────────────────────────────────────────────────────
# This is the master list of tech skills we detect.
# PhraseMatcher will scan resume + JD text for exact matches (case-insensitive).
# You can add more skills here anytime — just add to the right category.

SKILLS_TAXONOMY = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c",
        "go", "golang", "rust", "kotlin", "swift", "ruby", "php", "scala",
        "r", "matlab", "dart", "bash", "shell", "sql", "html", "css",
    ],
    "Frameworks & Libraries": [
        "react", "reactjs", "react.js", "angular", "vue", "vuejs", "next.js",
        "nextjs", "node.js", "nodejs", "express", "expressjs", "django",
        "flask", "fastapi", "spring", "spring boot", "springboot",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "matplotlib", "opencv", "huggingface",
        "langchain", "llamaindex", "llama-index", "streamlit",
        "tailwindcss", "tailwind", "bootstrap", "redux",
    ],
    "Databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite", "redis",
        "elasticsearch", "cassandra", "dynamodb", "firebase", "supabase",
        "chromadb", "pinecone", "faiss", "neo4j", "oracle",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "github actions", "ci/cd",
        "linux", "nginx", "apache", "heroku", "vercel", "netlify",
    ],
    "AI & ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "rag", "retrieval augmented generation",
        "llm", "large language model", "generative ai", "gen ai",
        "transformers", "bert", "gpt", "embeddings", "vector search",
        "fine-tuning", "prompt engineering", "langchain", "llamaindex",
        "reinforcement learning", "neural network", "cnn", "rnn", "lstm",
        "random forest", "xgboost", "regression", "classification",
    ],
    "Tools & Practices": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "postman", "rest api", "restful", "graphql", "grpc", "websocket",
        "agile", "scrum", "microservices", "system design",
        "data structures", "algorithms", "object oriented", "oop",
        "unit testing", "pytest", "jest", "selenium",
        "figma", "vs code", "linux", "unix",
    ],
}

# Flatten all skills into one list for matching
ALL_SKILLS = []
for category_skills in SKILLS_TAXONOMY.values():
    ALL_SKILLS.extend(category_skills)

# Remove duplicates while preserving order
ALL_SKILLS = list(dict.fromkeys(ALL_SKILLS))


# ── Education Keywords ────────────────────────────────────────────────────────

EDUCATION_KEYWORDS = {
    "PhD":        ["phd", "ph.d", "doctorate", "doctoral"],
    "Masters":    ["master", "m.tech", "m.e.", "mtech", "msc", "m.sc", "mba", "m.s"],
    "Bachelors":  ["bachelor", "b.tech", "btech", "b.e.", "be", "bsc", "b.sc", "b.s", "undergraduate"],
    "Diploma":    ["diploma", "polytechnic"],
}

# ── Seniority Keywords ────────────────────────────────────────────────────────

SENIORITY_KEYWORDS = {
    "Fresher":  ["fresher", "fresh graduate", "entry level", "entry-level", "intern", "0 year", "0-1 year"],
    "Junior":   ["junior", "associate", "1 year", "2 year", "1-2 year", "sde-1", "sde1"],
    "Mid":      ["mid", "3 year", "4 year", "3-5 year", "sde-2", "sde2"],
    "Senior":   ["senior", "lead", "5 year", "6 year", "7 year", "5+ year", "principal"],
}


# ── Core Extractor ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and clean text for matching."""
    return text.lower().strip()


def extract_skills(text: str) -> List[str]:
    """
    Extract tech skills from any text using exact phrase matching.

    How it works:
      1. Lowercase the text
      2. For each skill in our taxonomy, check if it appears in the text
      3. Return all matches

    Why not use spaCy NER here?
      spaCy's built-in NER (Named Entity Recognition) finds people, places,
      organisations — not tech skills. For tech skills we use our own
      vocabulary (SKILLS_TAXONOMY) with simple string matching.
      This is called a "gazetteer" approach — common in production NLP systems.
    """
    text_lower = _normalize(text)
    found = []
    for skill in ALL_SKILLS:
        # Use word boundary matching so "r" doesn't match inside "react"
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def extract_years_of_experience(text: str) -> int:
    """
    Extract years of experience from text using regex patterns.

    Handles patterns like:
      "3 years of experience"
      "5+ years"
      "2-3 years"
      "over 4 years"
    """
    text_lower = _normalize(text)

    # Pattern: number followed by "year"
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'(\d+)\s*-\s*\d+\s*years?',
        r'over\s+(\d+)\s*years?',
        r'(\d+)\+\s*years?',
    ]

    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            try:
                years_found.append(int(m))
            except ValueError:
                pass

    # Return the maximum years found (most relevant experience)
    return max(years_found) if years_found else 0


def extract_education(text: str) -> str:
    """
    Detect the highest education level mentioned in the text.
    Returns: PhD / Masters / Bachelors / Diploma / Not specified
    """
    text_lower = _normalize(text)
    # Check in order from highest to lowest
    for level, keywords in EDUCATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return "Not specified"


def extract_seniority(text: str) -> str:
    """
    Guess seniority level from text keywords.
    Returns: Fresher / Junior / Mid / Senior
    """
    text_lower = _normalize(text)
    for level, keywords in SENIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return "Fresher"


def get_skill_category(skill: str) -> str:
    """Return which category a skill belongs to."""
    for category, skills in SKILLS_TAXONOMY.items():
        if skill in skills:
            return category
    return "Other"


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_skills(resume_text: str, jd_text: str) -> dict:
    """
    Full NLP analysis of resume vs job description.

    Returns a dict with:
      - resume_skills: list of skills found in resume
      - jd_skills: list of skills required in JD
      - matched_skills: skills in both resume AND JD (strengths)
      - missing_skills: skills in JD but NOT in resume (gaps)
      - extra_skills: skills in resume but NOT in JD (bonus)
      - overlap_pct: percentage of JD skills covered by resume
      - years_experience: extracted from resume text
      - education: degree level detected
      - seniority: guessed seniority level
      - skill_breakdown: skills grouped by category
    """

    # Extract skills from both texts
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    # Set operations for gap analysis
    resume_set  = set(resume_skills)
    jd_set      = set(jd_skills)

    matched_skills = sorted(resume_set & jd_set)      # in both
    missing_skills = sorted(jd_set - resume_set)       # JD needs, resume lacks
    extra_skills   = sorted(resume_set - jd_set)       # resume has, JD doesn't ask for

    # Overlap percentage — how much of the JD's skill requirement you cover
    if jd_skills:
        overlap_pct = round(len(matched_skills) / len(jd_set) * 100, 1)
    else:
        overlap_pct = 0.0

    # Extract other resume metadata
    years_exp  = extract_years_of_experience(resume_text)
    education  = extract_education(resume_text)
    seniority  = extract_seniority(resume_text + " " + jd_text)

    # Group matched and missing skills by category for the UI
    skill_breakdown = {}
    for category in SKILLS_TAXONOMY:
        cat_matched  = [s for s in matched_skills if get_skill_category(s) == category]
        cat_missing  = [s for s in missing_skills if get_skill_category(s) == category]
        if cat_matched or cat_missing:
            skill_breakdown[category] = {
                "matched": cat_matched,
                "missing": cat_missing,
            }

    return {
        "resume_skills":   resume_skills,
        "jd_skills":       jd_skills,
        "matched_skills":  matched_skills,
        "missing_skills":  missing_skills,
        "extra_skills":    extra_skills,
        "overlap_pct":     overlap_pct,
        "years_experience": years_exp,
        "education":       education,
        "seniority":       seniority,
        "skill_breakdown": skill_breakdown,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
# Run this file directly to test: python src/nlp_engine.py

if __name__ == "__main__":
    sample_resume = """
    B.Tech Computer Science, 2023.
    Skills: Python, React, Node.js, MongoDB, REST API, Git, Docker.
    Built a face recognition attendance system using OpenCV and deep learning.
    2 years of experience in backend development.
    """

    sample_jd = """
    We are looking for a Backend Engineer with experience in Python, FastAPI,
    PostgreSQL, Docker, Kubernetes, and REST APIs. Knowledge of Redis and
    microservices is a plus. 2-3 years of experience required.
    """

    result = analyze_skills(sample_resume, sample_jd)

    print("Resume skills:",  result["resume_skills"])
    print("JD skills:",      result["jd_skills"])
    print("Matched:",        result["matched_skills"])
    print("Missing:",        result["missing_skills"])
    print("Overlap:",        result["overlap_pct"], "%")
    print("Experience:",     result["years_experience"], "years")
    print("Education:",      result["education"])
    print("Seniority:",      result["seniority"])
