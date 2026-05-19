"""Inputs resolver for the analysis.propagation_video phase.

Slice 3 of `analysis_propagation_video_plan.md`. Walks the recon-stage
output tree for a given ``(dataset, well, unit_id)`` and locates the
three artifacts the per-unit video renderer needs (per the slice-1
archeology audit at `dev/notes/refs/propagation_video_audit.md`):

- Merged template: ``<recon_outputs>/cache/templates/merged/unit_<id>/merged_template.npy``
  (v2; falls back to ``merged_contributing_template.npy`` for the legacy
  pre-concat-rip-out layout).
- Channel locations: same directory, ``merged_channel_locations.npy`` /
  ``merged_contributing_channel_locations.npy``.
- GTR pickle: ``<recon_outputs>/units/<unit_id>/gtr.pkl`` (the
  ``axon_velocity`` graph-tree-representation produced by the recon
  stage's ``axon_velocity_gtrs`` phase).

Slice 4 calls this resolver, loads the artifacts, and passes them
through to the per-unit video render logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....output_paths import compute_mea_analysis_output_dir


class PropagationVideoInputsMissing(FileNotFoundError):
	"""Raised when a unit's required propagation_video inputs aren't on disk."""


@dataclass(frozen=True)
class PropagationVideoInputs:
	"""Resolved per-unit input paths for the propagation_video phase."""

	dataset_index: int
	well_id: str
	unit_id: int
	well_out_dir: Path
	recon_output_dir: Path
	# Merged template + channel locations (v2 paths first, legacy fallback).
	merged_template_npy: Path
	merged_locations_npy: Path
	merged_meta_json: Path
	# Per-unit GTR pickle written by recon stage's axon_velocity_gtrs phase.
	gtr_pkl: Path
	gtr_json: Path

	def as_dict(self) -> dict[str, Any]:
		return {
			"dataset_index": int(self.dataset_index),
			"well_id": str(self.well_id),
			"unit_id": int(self.unit_id),
			"well_out_dir": str(self.well_out_dir),
			"recon_output_dir": str(self.recon_output_dir),
			"merged_template_npy": str(self.merged_template_npy),
			"merged_locations_npy": str(self.merged_locations_npy),
			"merged_meta_json": str(self.merged_meta_json),
			"gtr_pkl": str(self.gtr_pkl),
			"gtr_json": str(self.gtr_json),
		}


def _unit_dir_candidates(merged_units_dir: Path, unit_id: int) -> list[Path]:
	"""Candidate per-unit subdirectory names the recon stage might use.

	Mirrors ``stages/reconstruct/core/reconstruct.py``'s fallback chain:
	``unit_<id>`` (preferred) → ``<id:04d>`` → ``<id>`` (legacy).
	"""

	try:
		uid = int(unit_id)
	except Exception:
		uid = unit_id  # type: ignore[assignment]
	return [
		merged_units_dir / f"unit_{uid}",
		merged_units_dir / f"{int(uid):04d}",
		merged_units_dir / str(uid),
	]


def _resolve_merged_unit_dir(merged_units_dir: Path, unit_id: int) -> Path | None:
	for candidate in _unit_dir_candidates(merged_units_dir, unit_id):
		if candidate.is_dir():
			return candidate
	return None


def _resolve_template_files(unit_dir: Path) -> tuple[Path, Path, Path] | None:
	"""Pick the v2 layout if present, else legacy. Returns
	``(template_npy, locations_npy, meta_json)`` or None when neither
	exists.
	"""

	v2_tmpl = unit_dir / "merged_template.npy"
	v2_locs = unit_dir / "merged_channel_locations.npy"
	v2_meta = unit_dir / "unit_templates_summary.json"
	if v2_tmpl.is_file() and v2_locs.is_file():
		return v2_tmpl, v2_locs, v2_meta

	legacy_tmpl = unit_dir / "merged_contributing_template.npy"
	legacy_locs = unit_dir / "merged_contributing_channel_locations.npy"
	legacy_meta = unit_dir / "merged_contributing_template_meta.json"
	if legacy_tmpl.is_file() and legacy_locs.is_file():
		return legacy_tmpl, legacy_locs, legacy_meta
	return None


def resolve_propagation_video_inputs(
	*,
	dataset_index: int,
	well_id: str,
	unit_id: int,
	h5_path: str | Path,
	mea_output_root: str | Path,
	recon_output_rel_root: str = "recon_outputs",
	require_exists: bool = True,
) -> PropagationVideoInputs:
	"""Resolve every input path the propagation_video renderer needs.

	Arguments
	---------
	dataset_index, well_id, unit_id
	    Identify the (dataset, well, unit) tuple.
	h5_path
	    The dataset's raw h5 path; ``compute_mea_analysis_output_dir``
	    parses it to derive the well_out_dir.
	mea_output_root
	    The ``output_root`` from the data config (analyzed_data root).
	recon_output_rel_root
	    Canonical relpath under the well dir. Default
	    ``recon_outputs`` matches the recon-stage convention.
	require_exists
	    When True (default), raise ``PropagationVideoInputsMissing``
	    listing every missing input. When False, return the resolved
	    paths regardless — useful for dry-run wiring confirmation.
	"""

	well_out_dir = compute_mea_analysis_output_dir(
		output_root=Path(mea_output_root),
		data_file=Path(h5_path),
		well=str(well_id),
	)
	recon_output_dir = (well_out_dir / str(recon_output_rel_root)).resolve()
	merged_units_dir = recon_output_dir / "cache" / "templates" / "merged"
	unit_dir_merged = _resolve_merged_unit_dir(merged_units_dir, int(unit_id))
	# When no per-unit dir exists, point at the preferred candidate
	# anyway so the missing-inputs error message is actionable.
	if unit_dir_merged is None:
		unit_dir_merged = _unit_dir_candidates(merged_units_dir, int(unit_id))[0]
	template_files = _resolve_template_files(unit_dir_merged)
	if template_files is None:
		merged_template_npy = unit_dir_merged / "merged_template.npy"
		merged_locations_npy = unit_dir_merged / "merged_channel_locations.npy"
		merged_meta_json = unit_dir_merged / "unit_templates_summary.json"
	else:
		merged_template_npy, merged_locations_npy, merged_meta_json = template_files

	# Per-unit GTR from the recon stage's axon_velocity_gtrs phase.
	# Recon writes to ``<recon_outputs>/units/<unit_id_zero_padded>/`` by
	# default — see PerUnitOutputsConfig.unit_reldir convention.
	try:
		uid_str = f"{int(unit_id):04d}"
	except Exception:
		uid_str = str(unit_id)
	gtr_unit_dir = recon_output_dir / "units" / uid_str
	gtr_pkl = gtr_unit_dir / "gtr.pkl"
	gtr_json = gtr_unit_dir / "gtr.json"

	resolved = PropagationVideoInputs(
		dataset_index=int(dataset_index),
		well_id=str(well_id),
		unit_id=int(unit_id),
		well_out_dir=well_out_dir,
		recon_output_dir=recon_output_dir,
		merged_template_npy=merged_template_npy,
		merged_locations_npy=merged_locations_npy,
		merged_meta_json=merged_meta_json,
		gtr_pkl=gtr_pkl,
		gtr_json=gtr_json,
	)

	if require_exists:
		missing: list[str] = []
		suggestions: list[str] = []
		if not merged_template_npy.is_file():
			missing.append(f"merged_template_npy: {merged_template_npy}")
			suggestions.append(
				"Run `axon-recon stages reconstruct` first to produce merged "
				"per-unit templates."
			)
		if not merged_locations_npy.is_file():
			missing.append(f"merged_locations_npy: {merged_locations_npy}")
		if not gtr_pkl.is_file():
			missing.append(f"gtr_pkl: {gtr_pkl}")
			suggestions.append(
				"Run `axon-recon stages reconstruct.axon_velocity_gtrs` first "
				"to produce the per-unit GTR object the video renderer needs."
			)
		if missing:
			suggestion_block = (
				"\n  " + "\n  ".join(suggestions) if suggestions else ""
			)
			raise PropagationVideoInputsMissing(
				f"propagation_video: unit (dataset={dataset_index}, "
				f"well={well_id!r}, unit_id={unit_id}) is missing required inputs:\n  "
				+ "\n  ".join(missing)
				+ suggestion_block
			)

	return resolved


def discover_unit_ids_for_target(
	*,
	well_id: str,
	h5_path: str | Path,
	mea_output_root: str | Path,
	recon_output_rel_root: str = "recon_outputs",
) -> list[int]:
	"""Discover the per-unit subdirectories the recon stage produced for
	this target. Mirrors ``stages/reconstruct/runner.py:_discover_unit_ids``
	so slice 7's orchestrator can fan out without depending on the
	recon module directly.

	Returns sorted unit_id integers. Subdirectories that don't parse as
	integers (e.g. ``reports/`` / ``cache/``) are skipped.
	"""

	well_out_dir = compute_mea_analysis_output_dir(
		output_root=Path(mea_output_root),
		data_file=Path(h5_path),
		well=str(well_id),
	)
	merged_units_dir = (
		well_out_dir / str(recon_output_rel_root) / "cache" / "templates" / "merged"
	).resolve()
	if not merged_units_dir.is_dir():
		return []
	unit_ids: list[int] = []
	for entry in sorted(merged_units_dir.iterdir()):
		if not entry.is_dir():
			continue
		token = entry.name
		if token.startswith("unit_"):
			token = token.split("unit_", 1)[1]
		try:
			unit_ids.append(int(token))
		except Exception:
			continue
	return sorted(unit_ids)


__all__ = [
	"PropagationVideoInputs",
	"PropagationVideoInputsMissing",
	"discover_unit_ids_for_target",
	"resolve_propagation_video_inputs",
]
