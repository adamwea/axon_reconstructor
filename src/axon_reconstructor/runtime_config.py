from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RuntimeConfig:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload if isinstance(payload, dict) else {}

    @classmethod
    def load(cls, path: str | Path | None) -> "RuntimeConfig":
        if path is None:
            return cls({})
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Runtime config not found: {p}")

        raw = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()
        if suffix == ".json":
            payload = json.loads(raw)
        elif suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except Exception as e:
                raise RuntimeError("YAML runtime config requires PyYAML (`pip install pyyaml`).") from e
            payload = yaml.safe_load(raw)
        else:
            try:
                payload = json.loads(raw)
            except Exception:
                try:
                    import yaml  # type: ignore[import-not-found]
                except Exception as e:
                    raise RuntimeError(
                        f"Unsupported runtime config extension for {p}. Use .json/.yml/.yaml."
                    ) from e
                payload = yaml.safe_load(raw)

        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("Runtime config root must be a mapping/object")
        return cls(payload)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = dict(base)
        for key, value in override.items():
            if (
                key in out
                and isinstance(out.get(key), dict)
                and isinstance(value, dict)
            ):
                out[key] = RuntimeConfig._deep_merge(dict(out[key]), value)
            else:
                out[key] = value
        return out

    @classmethod
    def load_multiple(cls, paths: list[str | Path]) -> "RuntimeConfig":
        merged: dict[str, Any] = {}
        for path in list(paths):
            cfg = cls.load(path)
            merged = cls._deep_merge(merged, dict(cfg._payload))
        return cls(merged)

    def has(self, path: str) -> bool:
        marker = object()
        return self.get(path, marker) is not marker

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._payload
        for part in str(path).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_bool(self, path: str, default: bool | None = None) -> bool | None:
        value = self.get(path, None)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean at config path '{path}': {value!r}")

    def get_str(self, path: str, default: str | None = None) -> str | None:
        value = self.get(path, None)
        if value is None:
            return default
        token = str(value).strip()
        return token if token != "" else default

    def get_int(self, path: str, default: int | None = None) -> int | None:
        value = self.get(path, None)
        if value is None:
            return default
        try:
            return int(value)
        except Exception as e:
            raise ValueError(f"Invalid integer at config path '{path}': {value!r}") from e

    def get_float(self, path: str, default: float | None = None) -> float | None:
        value = self.get(path, None)
        if value is None:
            return default
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"Invalid float at config path '{path}': {value!r}") from e

    def get_path(self, path: str, default: str | Path | None = None) -> Path | None:
        value = self.get(path, None)
        if value is None:
            if default is None:
                return None
            return Path(default).expanduser().resolve()
        token = str(value).strip()
        if token == "":
            return None
        return Path(token).expanduser().resolve()

    def get_int_or_unlimited(self, path: str, default: int | None = None) -> int | None:
        value = self.get(path, None)
        if value is None:
            return default
        token = str(value).strip().lower()
        if token in {"", "none", "null", "all", "unlimited", "inf", "infinite"}:
            return -1
        try:
            return int(token)
        except Exception as e:
            raise ValueError(f"Invalid int-or-unlimited token at config path '{path}': {value!r}") from e
