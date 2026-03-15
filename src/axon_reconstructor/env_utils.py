#!/usr/bin/env python3
"""Shared env/config helpers for the project-local debug scripts.

Goal
----
All debug entrypoints in this folder should:
  1) Load a single env file (default: ./debug.env)
  2) Resolve defaults from environment variables (env file populates os.environ)
  3) Let CLI args override env values

We intentionally avoid external deps (e.g. python-dotenv).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


LEGACY_ENV_FILENAME = "debug.env"
DEFAULT_RUNTIME_ENV_FILENAME = "runtime.env"
DEFAULT_DATASET_ENV_FILENAME = "dataset.env"


class DebugEnvError(RuntimeError):
    pass


def default_env_path(*, script_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the default env file path.

    If script_path is provided, resolves relative to that file's directory.
    """

    if script_path is None:
        return Path(LEGACY_ENV_FILENAME)
    return Path(script_path).resolve().parent / LEGACY_ENV_FILENAME


def default_env_paths(*, script_path: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return default env file paths.

    Preferred (split) layout:
      - runtime.env   (pipeline knobs)
      - dataset.env   (paths + dataset selection)

    Backwards-compatible layout:
      - debug.env

    If one or both of the split env files exist next to the script, returns
    them (in load order). Otherwise returns [debug.env].
    """

    base_dir = Path(script_path).resolve().parent if script_path is not None else Path(".")
    legacy_env = base_dir / LEGACY_ENV_FILENAME

    # Prefer the legacy single-file config if it exists. This keeps the common
    # workflow simple and avoids accidentally switching behavior just because a
    # runtime.env or dataset.env file happens to be present.
    if legacy_env.exists():
        return [legacy_env]

    runtime_env = base_dir / DEFAULT_RUNTIME_ENV_FILENAME
    dataset_env = base_dir / DEFAULT_DATASET_ENV_FILENAME

    out: list[Path] = []
    if runtime_env.exists():
        out.append(runtime_env)
    if dataset_env.exists():
        out.append(dataset_env)

    if out:
        return out
    return [default_env_path(script_path=script_path)]


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _strip_inline_comment(value: str) -> str:
    """Strip a trailing inline comment.

    We treat a `#` as a comment delimiter only if it is preceded by whitespace.
    This keeps values like `foo#bar` intact.

    Examples:
      "0.75  # comment" -> "0.75"
      "relative" -> "relative"
    """

    import re

    # Don't strip comments from quoted strings.
    s = value.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s
    return re.split(r"\s+#", s, maxsplit=1)[0].strip()


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE env file.

    Rules:
    - blank lines ignored
    - lines starting with # ignored
    - inline comments are not supported (keep values simple)
    - surrounding single/double quotes are stripped
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    out: dict[str, str] = {}
    for i, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DebugEnvError(f"Invalid line {i} in {path}: expected KEY=VALUE")
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            raise DebugEnvError(f"Invalid line {i} in {path}: empty key")
        out[key] = _strip_quotes(_strip_inline_comment(v))
    return out


def load_env_file_into_os(*, env_file: Path, override_existing: bool = False) -> dict[str, str]:
    """Load env_file and set os.environ for any missing keys.

    Returns the parsed key/value dict.
    """

    parsed = parse_env_file(Path(env_file))
    for k, v in parsed.items():
        if (not override_existing) and (k in os.environ):
            continue
        # Treat empty values as "unset".
        if str(v).strip() == "":
            continue
        os.environ[k] = str(v)
    return parsed


def load_env_files_into_os(*, env_files: list[Path], override_existing: bool = False) -> dict[str, str]:
        """Load multiple env files into os.environ.

        Later files override earlier ones only if override_existing=True.
        For typical usage, keep override_existing=False and rely on precedence:
            shell env > env files > defaults.

        Returns merged parsed dict (later files win in the returned mapping).
        """

        merged: dict[str, str] = {}
        for p in env_files:
                merged.update(load_env_file_into_os(env_file=Path(p), override_existing=override_existing))
        return merged


def _env_get(key: str) -> str | None:
    v = os.environ.get(key)
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    return v


def env_str(key: str, *, default: str | None = None) -> str | None:
    v = _env_get(key)
    return default if v is None else v


def parse_typed_value(raw: str) -> Any:
    """Parse a simple env scalar into a typed Python value.

    Supports:
    - JSON literals/numbers (e.g. 1, 0.75, true, false)
    - "none"/"null" => None
    - "yes"/"no"/"on"/"off" => bool
    - otherwise returns the original string

    Notes:
    - For strings that must round-trip through JSON, prefer the JSON override mechanism.
    """

    import json

    s = str(raw).strip()
    if s == "":
        return None

    lowered = s.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"yes", "y", "on", "true"}:
        return True
    if lowered in {"no", "n", "off", "false"}:
        return False

    # Try JSON for numbers/bools/null.
    try:
        # json.loads is strict about lowercase true/false/null, so normalize first.
        if lowered in {"true", "false", "null"}:
            return json.loads(lowered)
        return json.loads(s)
    except Exception:
        return s


def env_typed(key: str, *, default: Any | None = None) -> Any:
    """Read an env var and parse it as a typed scalar."""

    v = _env_get(key)
    if v is None:
        return default
    return parse_typed_value(v)


def env_required_str(key: str) -> str:
    v = _env_get(key)
    if v is None:
        raise DebugEnvError(f"Missing required env var: {key}. Set it in debug.env or pass CLI args.")
    return v


def env_path(key: str, *, default: Path | None = None) -> Path | None:
    v = _env_get(key)
    return default if v is None else Path(v)


def env_required_path(key: str) -> Path:
    return Path(env_required_str(key))


def env_bool(key: str, *, default: bool = False) -> bool:
    v = _env_get(key)
    if v is None:
        return bool(default)
    return v.lower() in {"1", "true", "yes", "y", "on"}


def env_int(key: str, *, default: int | None = None) -> int | None:
    v = _env_get(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception as e:
        raise DebugEnvError(f"Invalid int for {key}={v!r}") from e


def env_float(key: str, *, default: float | None = None) -> float | None:
    v = _env_get(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception as e:
        raise DebugEnvError(f"Invalid float for {key}={v!r}") from e


def env_int_list(key: str) -> list[int] | None:
    v = _env_get(key)
    if v is None:
        return None
    out: list[int] = []
    for part in v.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def merge_json_overrides(base: dict[str, Any], *, json_str: str | None = None, json_path: Path | None = None) -> dict[str, Any]:
    """Merge optional JSON overrides into a dict (returns a new dict)."""

    import json

    out = dict(base)
    if json_path is not None:
        out.update(json.loads(Path(json_path).read_text(encoding="utf-8")))
    if json_str is not None:
        out.update(json.loads(str(json_str)))
    return out
