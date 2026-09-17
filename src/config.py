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


def get_secret(name: str, default: str = "") -> str:
    """
    Read a setting, checking in order:
      1. Environment variables — where python-dotenv puts your local .env, and
         where Streamlit Cloud mirrors top-level secrets.
      2. st.secrets — fallback for setups where that mirroring doesn't happen.
      3. The default you passed in.

    Always returns a stripped string, never None, so callers can use `if not x`.
    """
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets[name]
        if value:
            return str(value).strip()
    except Exception:
        # No Streamlit runtime, no secrets file, or no such key. All fine —
        # this is a fallback path, not the main one.
        pass

    return default


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
