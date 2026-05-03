from __future__ import annotations

import os
from pathlib import Path


def _append_text_line_locked(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _stale_recovery_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.stale")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale.{suffix}")
        suffix += 1
    return candidate


def _recover_unwritable_existing_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if not os.access(path.parent, os.W_OK | os.X_OK):
        return False
    path.rename(_stale_recovery_path(path))
    return True


def append_text_line(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _append_text_line_locked(path, text)
    except PermissionError:
        # Recover from stale root-owned logs left by older container runs when the
        # current user owns the parent log directory and can safely replace them.
        if not _recover_unwritable_existing_file(path):
            raise
        _append_text_line_locked(path, text)