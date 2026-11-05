"""Vercel entrypoint: exposes the FastAPI instance as `app` (Vercel looks for app.py at the project root)."""
import os

# Vercel's filesystem is read-only except /tmp; matplotlib needs a writable cache dir for the PDF charts.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.api import app  # noqa: E402,F401
