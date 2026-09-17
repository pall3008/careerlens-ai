"""
Job Search Engine - Fetches LIVE job openings matched to the candidate's resume.

Sources:
1. Adzuna API (primary)   - free tier, needs ADZUNA_APP_ID + ADZUNA_APP_KEY
   (from .env locally, or the Secrets panel when deployed)
   Sign up free at: https://developer.adzuna.com/
2. Remotive API (fallback) - free, no key required, remote tech jobs only

Both return a normalized list of job dicts so the UI doesn't care which source
a listing came from:
    {title, company, location, salary, posted, url, source}
"""

import requests

try:
    from src.config import get_secret
except ImportError:            # when this file is run directly from inside src/
    from config import get_secret

# Read lazily, not at import time. On a deployed app the secrets may be set or
# changed after the module is first imported, and module-level constants would
# freeze the old (empty) values until a full reboot.

def _adzuna_id() -> str:
    return get_secret("ADZUNA_APP_ID")


def _adzuna_key() -> str:
    return get_secret("ADZUNA_APP_KEY")


def _adzuna_country() -> str:
    return get_secret("ADZUNA_COUNTRY", "in") or "in"   # "in" = India

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
REQUEST_TIMEOUT = 10


def _adzuna_url() -> str:
    return f"https://api.adzuna.com/v1/api/jobs/{_adzuna_country()}/search/1"


def _format_salary(lo, hi) -> str:
    if not lo and not hi:
        return "Not specified"
    if lo and hi:
        return f"₹{int(lo):,} - ₹{int(hi):,}"
    return f"₹{int(lo or hi):,}+"


def adzuna_configured() -> bool:
    return bool(_adzuna_id() and _adzuna_key())


def search_adzuna(query: str, location: str = "", results: int = 15) -> list:
    """Search Adzuna for jobs. Returns [] silently if not configured or on error."""
    if not adzuna_configured():
        return []

    params = {
        "app_id": _adzuna_id(),
        "app_key": _adzuna_key(),
        "what": query,
        "results_per_page": results,
        "content-type": "application/json",
        "sort_by": "date",
    }
    if location:
        params["where"] = location

    try:
        resp = requests.get(_adzuna_url(), params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    jobs = []
    for item in data.get("results", []):
        title = (item.get("title") or "").replace("<strong>", "").replace("</strong>", "")
        company = (item.get("company") or {}).get("display_name", "Unknown")
        loc = (item.get("location") or {}).get("display_name", "")
        jobs.append({
            "title": title,
            "company": company,
            "location": loc,
            "salary": _format_salary(item.get("salary_min"), item.get("salary_max")),
            "posted": (item.get("created") or "")[:10],
            "url": item.get("redirect_url", "#"),
            "source": "Adzuna",
        })
    return jobs


def search_remotive(query: str, results: int = 15) -> list:
    """Search Remotive (remote jobs, no API key needed). Returns [] on error."""
    try:
        resp = requests.get(REMOTIVE_URL, params={"search": query}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    jobs = []
    for item in data.get("jobs", [])[:results]:
        jobs.append({
            "title": item.get("title", ""),
            "company": item.get("company_name", "Unknown"),
            "location": item.get("candidate_required_location", "Remote"),
            "salary": item.get("salary") or "Not specified",
            "posted": (item.get("publication_date") or "")[:10],
            "url": item.get("url", "#"),
            "source": "Remotive",
        })
    return jobs


def get_job_matches(search_query: str, location: str = "", remote_only: bool = False, results: int = 15) -> dict:
    """
    Combines Adzuna (if configured) + Remotive results into one ranked list.
    Returns: {"jobs": [...], "sources_used": [...], "adzuna_configured": bool}
    """
    jobs = []
    sources_used = []

    if not remote_only:
        adzuna_jobs = search_adzuna(search_query, location, results=results)
        if adzuna_jobs:
            jobs.extend(adzuna_jobs)
            sources_used.append("Adzuna")

    remotive_jobs = search_remotive(search_query, results=results)
    if remotive_jobs:
        jobs.extend(remotive_jobs)
        sources_used.append("Remotive")

    return {
        "jobs": jobs,
        "sources_used": sources_used,
        "adzuna_configured": adzuna_configured(),
    }
