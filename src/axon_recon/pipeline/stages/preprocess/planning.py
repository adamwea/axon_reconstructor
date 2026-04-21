from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RawPreprocessPlan:
	"""Plan for building a concatenated recording for spikesorting."""

	h5_path: Path
	stream_id: str
	cfg_files: tuple[Path, ...]
	cfg_discovery_summary: dict[str, object]


_WELL_TOKEN_RE = re.compile(r"(?:^|[_\-])well(?P<well>\d{1,3})(?:$|[_\-])", re.IGNORECASE)
_STREAM_PREFIX_RE = re.compile(r"^(?P<stream>\d+)[_-]\d+$")
_STREAM_TOKEN_RE = re.compile(r"(?:^|[_\-])(?:stream|strm|s)(?P<stream>\d{1,3})(?:$|[_\-])", re.IGNORECASE)


def _normalize_well_id(token: str) -> str:
	return f"well{int(token):03d}"


def infer_cfg_well_id(cfg_path: Path) -> str | None:
	"""Best-effort attribution of a cfg filename to a well id."""

	stem = Path(cfg_path).stem
	token = stem.strip()
	if not token:
		return None

	well_match = _WELL_TOKEN_RE.search(token)
	if well_match is not None:
		return _normalize_well_id(well_match.group("well"))

	prefix_match = _STREAM_PREFIX_RE.match(token)
	if prefix_match is not None:
		return _normalize_well_id(prefix_match.group("stream"))

	stream_match = _STREAM_TOKEN_RE.search(token)
	if stream_match is not None:
		return _normalize_well_id(stream_match.group("stream"))

	return None


def summarize_cfg_files_by_well(*, cfg_files: Iterable[Path], stream_id: str) -> dict[str, object]:
	"""Build deterministic cfg attribution counts grouped by well id."""

	files = [Path(p) for p in cfg_files]
	per_well_counts: dict[str, int] = {}
	unmatched_cfgs: list[str] = []

	for cfg in files:
		inferred_well = infer_cfg_well_id(cfg)
		if inferred_well is None:
			unmatched_cfgs.append(str(cfg.name))
			continue
		per_well_counts[inferred_well] = int(per_well_counts.get(inferred_well, 0) + 1)

	per_well_sorted = {well: int(per_well_counts[well]) for well in sorted(per_well_counts.keys())}
	unmatched_sorted = sorted(unmatched_cfgs)
	current_stream_key = str(stream_id)
	current_stream_count = int(per_well_sorted.get(current_stream_key, 0))

	return {
		"attribution_mode": "filename_best_effort",
		"total_cfg_files": int(len(files)),
		"matched_cfg_files": int(len(files) - len(unmatched_sorted)),
		"unmatched_cfg_files": int(len(unmatched_sorted)),
		"unmatched_cfg_examples": unmatched_sorted[:20],
		"per_well_cfg_counts": per_well_sorted,
		"current_stream_id": current_stream_key,
		"current_stream_cfg_count": current_stream_count,
	}


def format_cfg_discovery_summary(summary: dict[str, object]) -> str:
	"""Human-readable summary for preprocess startup logs."""

	counts = summary.get("per_well_cfg_counts", {})
	counts = counts if isinstance(counts, dict) else {}
	if counts:
		parts = [f"{str(k)}={int(v)}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]
		per_well_str = ", ".join(parts)
	else:
		per_well_str = "none"

	unmatched = int(summary.get("unmatched_cfg_files", 0) or 0)
	current_stream_id = str(summary.get("current_stream_id", "unknown"))
	current_stream_cfg_count = int(summary.get("current_stream_cfg_count", 0) or 0)
	return (
		f"per_well=({per_well_str}); "
		f"unmatched={unmatched}; "
		f"current_stream={current_stream_id}:{current_stream_cfg_count}"
	)


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


def build_preprocess_plan(*, h5_path: Path, stream_id: str, cfg_files: Iterable[Path] | None = None) -> RawPreprocessPlan:
	h5_path = Path(h5_path).expanduser().resolve()
	if cfg_files is None:
		cfg_files = discover_cfg_files(h5_path)
	cfg_files_tuple = tuple(Path(p).expanduser().resolve() for p in cfg_files)
	cfg_discovery_summary = summarize_cfg_files_by_well(cfg_files=cfg_files_tuple, stream_id=stream_id)
	return RawPreprocessPlan(
		h5_path=h5_path,
		stream_id=stream_id,
		cfg_files=cfg_files_tuple,
		cfg_discovery_summary=cfg_discovery_summary,
	)


__all__ = [
	"RawPreprocessPlan",
	"discover_cfg_files",
	"infer_cfg_well_id",
	"summarize_cfg_files_by_well",
	"format_cfg_discovery_summary",
	"parse_cfg_channel_locations",
	"build_preprocess_plan",
]
