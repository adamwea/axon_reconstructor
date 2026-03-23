from __future__ import annotations

from typing import Any


def normalize_corner_location(raw: Any, *, default: str = "bottomright") -> str:
    """Normalize corner-location tokens for overlays/annotations."""

    v = str(raw or default).strip().lower().replace("_", " ")
    aliases = {
        "topleft": "topleft",
        "top left": "topleft",
        "topright": "topright",
        "top right": "topright",
        "bottomleft": "bottomleft",
        "bottom left": "bottomleft",
        "bottomright": "bottomright",
        "bottom right": "bottomright",
    }
    return aliases.get(v, str(default or "bottomright"))


def colorbar_axes_bounds(*, location: Any, length_fraction: float, pad_fraction: float) -> list[float]:
    """Return [left, bottom, width, height] for a figure-level vertical colorbar."""

    loc = normalize_corner_location(location, default="topright")
    length = min(0.95, max(0.05, float(length_fraction)))
    pad = min(0.25, max(0.0, float(pad_fraction)))
    width = 0.015

    if loc in {"topleft", "bottomleft"}:
        left = 0.06 + pad
    else:
        left = 0.93 - width - pad

    if loc in {"topleft", "topright"}:
        bottom = 0.97 - length - pad
    else:
        bottom = 0.03 + pad
    return [float(left), float(bottom), float(width), float(length)]
