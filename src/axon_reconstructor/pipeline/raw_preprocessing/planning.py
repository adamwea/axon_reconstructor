from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RawPreprocessPlan:
    """Plan for building a concatenated recording for spikesorting."""

    h5_path: Path
    stream_id: str
    cfg_files: tuple[Path, ...]


def discover_cfg_files(h5_path: Path) -> list[Path]:
    """Return `.cfg` files adjacent to an `.h5` file."""

    h5_path = Path(h5_path)
    folder = h5_path.parent
    return sorted(folder.glob("*.cfg"))


def parse_cfg_channel_locations(cfg_path: Path) -> dict:
    """Parse a Maxwell-style `.cfg` file (best effort)."""

    cfg_path = Path(cfg_path)
    raw = cfg_path.read_text(errors="replace")

    parser = configparser.ConfigParser()
    sections: dict[str, dict[str, str]] = {}
    try:
        parser.read_string(raw)
        for section in parser.sections():
            sections[section] = dict(parser.items(section))
    except configparser.Error:
        sections = {}

    return {"path": str(cfg_path), "sections": sections, "raw": raw}


def build_preprocess_plan(*, h5_path: Path, stream_id: str, cfg_files: Optional[Iterable[Path]] = None) -> RawPreprocessPlan:
    h5_path = Path(h5_path).expanduser().resolve()
    if cfg_files is None:
        cfg_files = discover_cfg_files(h5_path)
    cfg_files_tuple = tuple(Path(p).expanduser().resolve() for p in cfg_files)
    return RawPreprocessPlan(h5_path=h5_path, stream_id=stream_id, cfg_files=cfg_files_tuple)


__all__ = [
    "RawPreprocessPlan",
    "discover_cfg_files",
    "parse_cfg_channel_locations",
    "build_preprocess_plan",
]
