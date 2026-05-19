"""Slice 3 of analysis_propagation_video_plan: inputs resolver tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.stages.analysis.core.propagation_video_inputs import (
	PropagationVideoInputs,
	PropagationVideoInputsMissing,
	resolve_propagation_video_inputs,
)


_H5_TEMPLATE = "/raw/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


def _scaffold_well_outputs_v2(
	tmp_path: Path,
	*,
	chip: str = "M08073",
	well_id: str = "well000",
	date: str = "260224",
	run: str = "000001",
	unit_ids: tuple[int, ...] = (5,),
	include_gtr: bool = True,
	include_templates: bool = True,
) -> Path:
	"""Scaffold the v2 recon-output layout per the audit:

	  <output_root>/proj/<date>/<chip>/AxonTracking/<run>/<well_id>/
	    recon_outputs/
	      cache/templates/merged/unit_<id>/{merged_template.npy, merged_channel_locations.npy, unit_templates_summary.json}
	      units/<unit_id_zero_padded>/{gtr.pkl, gtr.json}
	"""

	output_root = tmp_path / "output_root"
	well_dir = (
		output_root
		/ "proj"
		/ date
		/ chip
		/ "AxonTracking"
		/ run
		/ well_id
	)
	recon_dir = well_dir / "recon_outputs"
	for unit_id in unit_ids:
		if include_templates:
			unit_tmpl_dir = recon_dir / "cache" / "templates" / "merged" / f"unit_{unit_id}"
			unit_tmpl_dir.mkdir(parents=True, exist_ok=True)
			(unit_tmpl_dir / "merged_template.npy").write_bytes(b"\x00")
			(unit_tmpl_dir / "merged_channel_locations.npy").write_bytes(b"\x00")
			(unit_tmpl_dir / "unit_templates_summary.json").write_text("{}", encoding="utf-8")
		if include_gtr:
			uid_str = f"{int(unit_id):04d}"
			gtr_unit_dir = recon_dir / "units" / uid_str
			gtr_unit_dir.mkdir(parents=True, exist_ok=True)
			(gtr_unit_dir / "gtr.pkl").write_bytes(b"\x00")
			(gtr_unit_dir / "gtr.json").write_text("{}", encoding="utf-8")
	return output_root


# --- happy path ---


def test_resolves_v2_paths_when_layout_exists(tmp_path: Path) -> None:
	output_root = _scaffold_well_outputs_v2(tmp_path, unit_ids=(5,))
	inputs = resolve_propagation_video_inputs(
		dataset_index=0,
		well_id="well000",
		unit_id=5,
		h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
		mea_output_root=output_root,
	)
	assert isinstance(inputs, PropagationVideoInputs)
	assert inputs.unit_id == 5
	assert inputs.merged_template_npy.is_file()
	assert inputs.merged_locations_npy.is_file()
	assert inputs.gtr_pkl.is_file()
	assert "merged_template.npy" in str(inputs.merged_template_npy)
	# Recon stage uses zero-padded `0005` for the per-unit GTR dir.
	assert "0005" in str(inputs.gtr_pkl)


# --- legacy fallback ---


def test_resolves_legacy_template_paths(tmp_path: Path) -> None:
	"""When `merged_template.npy` doesn't exist but the legacy
	`merged_contributing_template.npy` does, the resolver should pick
	the legacy layout. Mirrors `core/reconstruct.py:96-108`."""

	output_root = tmp_path / "output_root"
	well_dir = output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000"
	unit_tmpl_dir = well_dir / "recon_outputs" / "cache" / "templates" / "merged" / "unit_5"
	unit_tmpl_dir.mkdir(parents=True, exist_ok=True)
	(unit_tmpl_dir / "merged_contributing_template.npy").write_bytes(b"\x00")
	(unit_tmpl_dir / "merged_contributing_channel_locations.npy").write_bytes(b"\x00")
	gtr_dir = well_dir / "recon_outputs" / "units" / "0005"
	gtr_dir.mkdir(parents=True, exist_ok=True)
	(gtr_dir / "gtr.pkl").write_bytes(b"\x00")
	(gtr_dir / "gtr.json").write_text("{}", encoding="utf-8")

	inputs = resolve_propagation_video_inputs(
		dataset_index=0,
		well_id="well000",
		unit_id=5,
		h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
		mea_output_root=output_root,
	)
	assert inputs.merged_template_npy.name == "merged_contributing_template.npy"
	assert inputs.merged_locations_npy.name == "merged_contributing_channel_locations.npy"


# --- error paths ---


def test_missing_template_raises_with_actionable_suggestion(tmp_path: Path) -> None:
	output_root = _scaffold_well_outputs_v2(
		tmp_path, unit_ids=(5,), include_templates=False
	)
	with pytest.raises(PropagationVideoInputsMissing) as exc_info:
		resolve_propagation_video_inputs(
			dataset_index=0,
			well_id="well000",
			unit_id=5,
			h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
			mea_output_root=output_root,
		)
	msg = str(exc_info.value)
	assert "merged_template_npy" in msg
	assert "axon-recon stages reconstruct" in msg


def test_missing_gtr_raises_with_actionable_suggestion(tmp_path: Path) -> None:
	output_root = _scaffold_well_outputs_v2(
		tmp_path, unit_ids=(5,), include_gtr=False
	)
	with pytest.raises(PropagationVideoInputsMissing) as exc_info:
		resolve_propagation_video_inputs(
			dataset_index=0,
			well_id="well000",
			unit_id=5,
			h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
			mea_output_root=output_root,
		)
	msg = str(exc_info.value)
	assert "gtr_pkl" in msg
	assert "axon_velocity_gtrs" in msg


def test_require_exists_false_skips_validation(tmp_path: Path) -> None:
	"""Dry-run mode: return paths regardless of disk presence."""
	output_root = _scaffold_well_outputs_v2(
		tmp_path, unit_ids=(5,), include_gtr=False, include_templates=False
	)
	inputs = resolve_propagation_video_inputs(
		dataset_index=0,
		well_id="well000",
		unit_id=5,
		h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
		mea_output_root=output_root,
		require_exists=False,
	)
	# Returns even though files are missing.
	assert isinstance(inputs, PropagationVideoInputs)
	assert not inputs.merged_template_npy.is_file()
	assert not inputs.gtr_pkl.is_file()


def test_as_dict_round_trips_paths(tmp_path: Path) -> None:
	output_root = _scaffold_well_outputs_v2(tmp_path, unit_ids=(5,))
	inputs = resolve_propagation_video_inputs(
		dataset_index=0,
		well_id="well000",
		unit_id=5,
		h5_path=_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001"),
		mea_output_root=output_root,
	)
	d = inputs.as_dict()
	assert isinstance(d, dict)
	assert d["unit_id"] == 5
	assert d["dataset_index"] == 0
	assert d["well_id"] == "well000"
	# Every path field is a string.
	for key in (
		"well_out_dir",
		"recon_output_dir",
		"merged_template_npy",
		"merged_locations_npy",
		"gtr_pkl",
		"gtr_json",
	):
		assert isinstance(d[key], str)
