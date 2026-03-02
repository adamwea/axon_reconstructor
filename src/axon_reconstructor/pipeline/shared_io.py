from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def jsonable(x: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return x


def jsonable_sequence(xs: Any) -> list[Any] | None:
    if xs is None:
        return None
    try:
        return [jsonable(v) for v in list(xs)]
    except Exception:
        try:
            return [jsonable(xs)]
        except Exception:
            return None


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, np.ndarray):
            return x.ravel().tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]


def as_float_list(x: Any) -> list[float]:
    out: list[float] = []
    for value in as_list(x):
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def as_int_list(x: Any) -> list[int]:
    out: list[int] = []
    for value in as_list(x):
        try:
            out.append(int(value))
        except Exception:
            continue
    return out
