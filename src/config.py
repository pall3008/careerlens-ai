"""
Config — one place to read settings and pick writable paths.

Why this exists:
  The app runs in two very different places. Locally, keys come from .env and
  the project folder is writable. On a host like Streamlit Community Cloud
  there is no .env (keys come from the Secrets UI) and the checkout may be
  read-only. Rather than scattering that difference across modules, both cases
  are handled here once.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Loads .env when running locally. A no-op when the file doesn't exist.
load_dotenv()


def _streamlit_secrets():
    """Return st.secrets, or None if there's no Streamlit runtime / secrets file."""
    try:
        import streamlit as st

        # Touch it once so a missing secrets file raises here, not at the call site.
        _ = list(st.secrets.keys())
        return st.secrets
    except Exception:
        return None


def get_secret(name: str, default: str = "") -> str:
    """
    Read a setting, checking in order:
      1. Environment variables — where python-dotenv puts your local .env, and
         where Streamlit Cloud mirrors top-level secrets.
      2. st.secrets at the top level.
      3. st.secrets one level deep, in case the key was pasted under a
         [section] header in the Secrets box. That nests it, and a top-level
         lookup would otherwise miss it entirely.
      4. The default you passed in.

    Always returns a stripped string, never None, so callers can use `if not x`.
    """
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    secrets = _streamlit_secrets()
    if secrets is not None:
        try:
            if name in secrets and secrets[name]:
                return str(secrets[name]).strip()
        except Exception:
            pass

        # Nested one level: [some_section] \n GROQ_API_KEY = "..."
        try:
            for section in secrets.values():
                if hasattr(section, "keys") and name in section and section[name]:
                    return str(section[name]).strip()
        except Exception:
            pass

    return default


def available_secret_names() -> list:
    """
    Names (never values) of the secrets the app can currently see.

    Purely for error messages: when a key is missing, showing what IS present
    turns 'not found' into an obvious diagnosis — wrong spelling, wrong case,
    or nested under a section header.
    """
    names = []

    secrets = _streamlit_secrets()
    if secrets is not None:
        try:
            for key, value in secrets.items():
                if hasattr(value, "keys"):
                    names.extend(f"{key}.{sub}" for sub in value.keys())
                else:
                    names.append(key)
        except Exception:
            pass

    # Also surface relevant env vars, without ever revealing a value.
    for key in os.environ:
        if any(tag in key.upper() for tag in ("GROQ", "ADZUNA")) and key not in names:
            names.append(key)

    return sorted(names)


def writable_dir(preferred, fallback_name: str) -> Path:
    """
    Return a directory we can actually write to.

    Locally that's the project folder. If the checkout is read-only — which
    happens on some hosts — fall back to the system temp directory. Everything
    stored through this function is a rebuildable cache (vector store, model
    weights), so losing it on restart costs time, not data.
    """
    preferred = Path(preferred)
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / fallback_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
