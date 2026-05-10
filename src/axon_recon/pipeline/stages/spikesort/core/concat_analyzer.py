"""Build (or reuse) the canonical concat-level SortingAnalyzer.

This is the slice-2 implementation. Slice 3 wires bombcell_label onto it,
slice 4 wires merge_SLAy, slice 5 wires merge_si_auto / merge_unitmatch.
Today this lives alongside the existing per-phase analyzer-build helpers
in `runner.py`; those are removed in slices 3-5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FINGERPRINT_FILENAME = "sorter_output_fingerprint.json"


# Default extension set covers the union of what bombcell_label and the
# merge stage build today (see runner.py:1684 _ensure_merge_analyzer_extensions
# and runner.py:2778 _ensure_bombcell_metric_extensions). Slice 3-5 will
# narrow or extend this as needed.
DEFAULT_CONCAT_ANALYZER_EXTENSIONS: dict[str, dict[str, Any]] = {
	"random_spikes": {"max_spikes_per_unit": 500},
	"waveforms": {"ms_before": 1.0, "ms_after": 2.0},
	"templates": {},
	"noise_levels": {},
}


def _hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
	h = hashlib.sha256()
	with path.open("rb") as fh:
		for chunk in iter(lambda: fh.read(chunk_size), b""):
			h.update(chunk)
	return h.hexdigest()


def compute_sorter_output_fingerprint(sorter_output_dir: Path) -> dict[str, Any]:
	"""Compute a deterministic per-file sha256 fingerprint of sorter_output."""
	root = Path(sorter_output_dir).resolve()
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"concat_analyzer fingerprint: {root} not found")
	files = sorted(p for p in root.rglob("*") if p.is_file())
	per_file: dict[str, str] = {}
	combined = hashlib.sha256()
	total_bytes = 0
	for f in files:
		rel = f.relative_to(root).as_posix()
		digest = _hash_file(f)
		per_file[rel] = digest
		combined.update(rel.encode("utf-8"))
		combined.update(b"\x00")
		combined.update(digest.encode("ascii"))
		combined.update(b"\x00")
		total_bytes += f.stat().st_size
	return {
		"source_dir": str(root),
		"file_count": len(files),
		"total_bytes": int(total_bytes),
		"combined_sha256": combined.hexdigest(),
		"per_file_sha256": per_file,
	}


def _read_existing_fingerprint(analyzer_dir: Path) -> dict[str, Any] | None:
	path = Path(analyzer_dir) / FINGERPRINT_FILENAME
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None


def _write_fingerprint(analyzer_dir: Path, fingerprint: Mapping[str, Any]) -> Path:
	path = Path(analyzer_dir) / FINGERPRINT_FILENAME
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = dict(fingerprint)
	payload["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return path


def fingerprints_match(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
	"""Two fingerprints match iff combined_sha256 + file_count + total_bytes match."""
	if a is None or b is None:
		return False
	keys = ("combined_sha256", "file_count", "total_bytes")
	for k in keys:
		if a.get(k) != b.get(k):
			return False
	return True


def run_concat_analyzer_phase(
	*,
	sorter_output_dir: Path,
	recording: Any,
	sorting: Any,
	analyzer_dir: Path,
	extensions: Mapping[str, Mapping[str, Any]] | None = None,
	rebuild_on_sorter_output_change: bool = True,
	analyzer_format: str = "binary_folder",
	compute_sparsity: bool = True,
	n_jobs: int | None = None,
	create_sorting_analyzer_fn: Any | None = None,
	load_sorting_analyzer_fn: Any | None = None,
	logger: logging.Logger | None = None,
) -> dict[str, Any]:
	"""Build (or reuse) the canonical concat-level SortingAnalyzer.

	Behavior:
	- Compute the current sorter_output fingerprint (sha256 of every file).
	- If `analyzer_dir` exists AND its stored fingerprint matches the current
	  one AND `rebuild_on_sorter_output_change=True`, load the existing
	  analyzer and return ``{"rebuilt": False, ...}``.
	- Otherwise: wipe `analyzer_dir`, call create_sorting_analyzer with the
	  given recording + sorting, compute requested extensions, then write
	  the fingerprint file. Return ``{"rebuilt": True, ...}``.

	`create_sorting_analyzer_fn` / `load_sorting_analyzer_fn` are injection
	points for tests (and for callers that don't want this module to depend
	on `spikeinterface` at import time). When omitted, we import them from
	the live `spikeinterface` module on demand.
	"""
	analyzer_dir = Path(analyzer_dir).resolve()
	sorter_output_dir = Path(sorter_output_dir).resolve()
	current_fingerprint = compute_sorter_output_fingerprint(sorter_output_dir)

	stored_fingerprint = _read_existing_fingerprint(analyzer_dir)
	can_reuse = (
		bool(rebuild_on_sorter_output_change)
		and analyzer_dir.exists()
		and fingerprints_match(stored_fingerprint, current_fingerprint)
	)

	if can_reuse and load_sorting_analyzer_fn is None:
		try:
			import spikeinterface  # type: ignore

			load_sorting_analyzer_fn = getattr(spikeinterface, "load_sorting_analyzer", None)
		except Exception:
			load_sorting_analyzer_fn = None

	rebuilt = False
	rebuild_reason: str | None = None
	analyzer_obj: Any | None = None

	if can_reuse and callable(load_sorting_analyzer_fn):
		try:
			analyzer_obj = load_sorting_analyzer_fn(analyzer_dir)
			if logger is not None:
				logger.info(
					"concat_analyzer: reused existing analyzer at %s (fingerprint match)",
					str(analyzer_dir),
				)
		except Exception as exc:
			rebuild_reason = f"load_failed:{type(exc).__name__}"
			analyzer_obj = None
			rebuilt = True
	else:
		rebuilt = True
		if not analyzer_dir.exists():
			rebuild_reason = "analyzer_dir_missing"
		elif stored_fingerprint is None:
			rebuild_reason = "fingerprint_missing"
		elif not fingerprints_match(stored_fingerprint, current_fingerprint):
			rebuild_reason = "sorter_output_fingerprint_changed"
		elif not bool(rebuild_on_sorter_output_change):
			rebuild_reason = "rebuild_on_sorter_output_change_disabled"
		else:
			rebuild_reason = "load_api_unavailable"

	if rebuilt:
		if create_sorting_analyzer_fn is None:
			import spikeinterface  # type: ignore

			create_sorting_analyzer_fn = getattr(spikeinterface, "create_sorting_analyzer", None)
		if not callable(create_sorting_analyzer_fn):
			raise RuntimeError(
				"concat_analyzer: spikeinterface.create_sorting_analyzer is required to build the analyzer"
			)
		if analyzer_dir.exists():
			shutil.rmtree(analyzer_dir, ignore_errors=False)
		create_kwargs: dict[str, Any] = {
			"sorting": sorting,
			"recording": recording,
			"format": str(analyzer_format),
			"folder": analyzer_dir,
		}
		if not bool(compute_sparsity):
			create_kwargs["sparse"] = False
		if logger is not None:
			logger.info(
				"concat_analyzer: building analyzer at %s (reason=%s)",
				str(analyzer_dir),
				rebuild_reason or "rebuild_requested",
			)
		try:
			analyzer_obj = create_sorting_analyzer_fn(**create_kwargs)
		except TypeError:
			# Older spikeinterface versions may reject the `sparse` kwarg.
			if "sparse" not in create_kwargs:
				raise
			create_kwargs.pop("sparse", None)
			analyzer_obj = create_sorting_analyzer_fn(**create_kwargs)

	resolved_extensions: dict[str, dict[str, Any]] = {}
	if extensions is None:
		resolved_extensions = {k: dict(v) for k, v in DEFAULT_CONCAT_ANALYZER_EXTENSIONS.items()}
	else:
		for ext_name, ext_kwargs in extensions.items():
			resolved_extensions[str(ext_name)] = dict(ext_kwargs or {})

	computed_extensions: list[str] = []
	skipped_extensions: list[str] = []
	if rebuilt:
		# Compute every requested extension on a freshly-built analyzer.
		for ext_name, ext_kwargs in resolved_extensions.items():
			compute_kwargs = dict(ext_kwargs)
			if n_jobs is not None and "n_jobs" not in compute_kwargs:
				compute_kwargs["n_jobs"] = int(n_jobs)
			try:
				analyzer_obj.compute(ext_name, **compute_kwargs)
				computed_extensions.append(ext_name)
			except Exception as exc:
				if logger is not None:
					logger.warning(
						"concat_analyzer: failed to compute extension %s: %s",
						ext_name,
						exc,
					)
				skipped_extensions.append(f"{ext_name}:{type(exc).__name__}")
	else:
		# Re-used analyzer: assume every extension is already on disk.
		for ext_name in resolved_extensions.keys():
			if _analyzer_has_extension(analyzer_obj, ext_name):
				computed_extensions.append(ext_name)

	fingerprint_path = _write_fingerprint(analyzer_dir, current_fingerprint)

	return {
		"status": "ok",
		"analyzer_dir": str(analyzer_dir),
		"sorter_output_dir": str(sorter_output_dir),
		"format": str(analyzer_format),
		"rebuilt": bool(rebuilt),
		"rebuild_reason": rebuild_reason,
		"extensions_requested": list(resolved_extensions.keys()),
		"extensions_computed": list(computed_extensions),
		"extensions_skipped": list(skipped_extensions),
		"extension_count": len(computed_extensions),
		"fingerprint_path": str(fingerprint_path),
		"file_count": int(current_fingerprint["file_count"]),
		"total_bytes": int(current_fingerprint["total_bytes"]),
		"combined_sha256": str(current_fingerprint["combined_sha256"]),
	}


def _analyzer_has_extension(analyzer: Any, extension_name: str) -> bool:
	try:
		has_ext = getattr(analyzer, "has_extension", None)
		if callable(has_ext):
			return bool(has_ext(extension_name))
	except Exception:
		pass
	try:
		extensions = getattr(analyzer, "extensions", None)
		if extensions is not None and extension_name in extensions:
			return True
	except Exception:
		pass
	return False
