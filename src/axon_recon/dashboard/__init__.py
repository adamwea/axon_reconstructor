"""Dashboard package — Dash app over per-well analysis_outputs/ artifacts."""

from .app import build_app
from .data import load_all
from .discovery import iter_manifest_paths

__all__ = ["build_app", "iter_manifest_paths", "load_all"]
