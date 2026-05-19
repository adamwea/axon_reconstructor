"""Tests for analysis.core.unitmatch_groups."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.stages.analysis.core.unitmatch_groups import (
	SessionInputs,
	UnitmatchSessionInputMissing,
	discover_chip_well_groups,
	resolve_group_session_inputs,
	resolve_session_inputs,
)
from axon_recon.runtime_config import RuntimeConfig


# Canonical raw_data_h5_path shape: <project>/<YYMMDD>/<chip>/<scan_type>/<run>/data.raw.h5
_H5_TEMPLATE = "/pscratch/raw_data/Media_Density_T5/Media_Density_T5/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


def _data_cfg_with_datasets(*entries: dict) -> RuntimeConfig:
	return RuntimeConfig({"datasets": list(entries)})


def _ds(*, date: str, chip: str, well_ids: list[str], run: str = "000001") -> dict:
	return {
		"raw_data_h5_path": _H5_TEMPLATE.format(date=date, chip=chip, run=run),
		"DIV": int(date[-2:]),
		"wells": [{"well_id": w, "attributes": {}} for w in well_ids],
	}


def test_discover_chip_well_groups_groups_by_chip_and_well() -> None:
	# Three datasets across two chips and two wells. The function should
	# produce one group per (chip, well).
	cfg = _data_cfg_with_datasets(
		_ds(date="260224", chip="M08073", well_ids=["well000", "well001"], run="000001"),
		_ds(date="260226", chip="M08073", well_ids=["well000"], run="000002"),
		_ds(date="260224", chip="M99999", well_ids=["well000"], run="000003"),
	)

	groups = discover_chip_well_groups(cfg)
	assert groups == {
		("M08073", "well000"): [0, 1],
		("M08073", "well001"): [0],
		("M99999", "well000"): [2],
	}


def test_discover_chip_well_groups_skips_unparseable_h5() -> None:
	# A dataset whose h5 path doesn't match the canonical shape should
	# be silently dropped (no chip_id available).
	cfg = _data_cfg_with_datasets(
		_ds(date="260224", chip="M08073", well_ids=["well000"]),
		{"raw_data_h5_path": "/tmp/some/short/path.h5", "wells": [{"well_id": "well000"}]},
	)
	groups = discover_chip_well_groups(cfg)
	# Only the canonical-shape entry survives.
	assert ("M08073", "well000") in groups
	assert all(chip is not None for chip, _ in groups.keys())


def test_discover_chip_well_groups_from_well_metadata_directly() -> None:
	# When the caller already built a well_metadata lookup, the function
	# uses it directly.
	well_metadata = {
		(0, "well000"): {"chip_id": "M08073"},
		(1, "well000"): {"chip_id": "M08073"},
		(0, "well001"): {"chip_id": "M08073"},
		(2, "well000"): {"chip_id": "M99999"},
	}
	groups = discover_chip_well_groups(well_metadata=well_metadata)
	assert groups == {
		("M08073", "well000"): [0, 1],
		("M08073", "well001"): [0],
		("M99999", "well000"): [2],
	}


def test_discover_chip_well_groups_requires_one_input_source() -> None:
	with pytest.raises(ValueError, match="data_cfg or well_metadata"):
		discover_chip_well_groups()


def _scaffold_recon_outputs(tmp_path: Path, *, h5_relpath: str, well_id: str) -> Path:
	"""Create the per-well recon_outputs tree the resolver expects to see."""

	output_root = tmp_path / "output_root"
	well_out = output_root / Path(h5_relpath).parent / well_id
	recon = well_out / "recon_outputs"
	(recon / "synth_sorter_output").mkdir(parents=True, exist_ok=True)
	(recon / "cache" / "analyzers" / "segments").mkdir(parents=True, exist_ok=True)
	return output_root


def test_resolve_session_inputs_returns_canonical_paths(tmp_path: Path) -> None:
	# Set up a fake well output tree mirroring what kssynth + recon
	# produce, then verify the resolver finds them.
	h5_path = "/pscratch/raw_data/proj/proj/260224/M08073/AxonTracking/000001/data.raw.h5"
	# compute_mea_analysis_output_dir uses the last 5 path parts before the
	# data file → ``proj/260224/M08073/AxonTracking/000001/well000``.
	output_root = tmp_path / "output_root"
	per_well = output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000"
	(per_well / "recon_outputs" / "synth_sorter_output").mkdir(parents=True)
	(per_well / "recon_outputs" / "cache" / "analyzers" / "segments").mkdir(parents=True)

	resolved = resolve_session_inputs(
		dataset_index=0,
		well_id="well000",
		h5_path=h5_path,
		output_root=output_root,
	)
	assert isinstance(resolved, SessionInputs)
	assert resolved.sorter_output == (per_well / "recon_outputs" / "synth_sorter_output").resolve()
	assert resolved.analyzers == (per_well / "recon_outputs" / "cache" / "analyzers" / "segments").resolve()


def test_resolve_session_inputs_raises_when_synth_missing(tmp_path: Path) -> None:
	h5_path = "/pscratch/raw_data/proj/proj/260224/M08073/AxonTracking/000001/data.raw.h5"
	output_root = tmp_path / "output_root"
	per_well = output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000"
	# Only create the analyzers tree; synth is missing.
	(per_well / "recon_outputs" / "cache" / "analyzers" / "segments").mkdir(parents=True)

	with pytest.raises(UnitmatchSessionInputMissing) as exc:
		resolve_session_inputs(
			dataset_index=0,
			well_id="well000",
			h5_path=h5_path,
			output_root=output_root,
		)
	assert "reconstruct.kssynth" in str(exc.value)
	assert "sorter_output" in str(exc.value)


def test_resolve_session_inputs_skip_existence_check_with_flag(tmp_path: Path) -> None:
	# When require_exists=False, the resolver returns paths regardless.
	resolved = resolve_session_inputs(
		dataset_index=0,
		well_id="well000",
		h5_path="/some/path/proj/proj/260224/M08073/AxonTracking/000001/data.raw.h5",
		output_root=tmp_path / "nonexistent",
		require_exists=False,
	)
	assert resolved.well_id == "well000"
	# Path was resolved (not necessarily extant).
	assert "synth_sorter_output" in str(resolved.sorter_output)


def test_resolve_group_session_inputs_handles_full_group(tmp_path: Path) -> None:
	# 2 datasets sharing (chip, well); both have synth + analyzers on disk.
	cfg = _data_cfg_with_datasets(
		_ds(date="260224", chip="M08073", well_ids=["well000"], run="000001"),
		_ds(date="260226", chip="M08073", well_ids=["well000"], run="000002"),
	)
	output_root = tmp_path / "output_root"
	for run in ("000001", "000002"):
		date = "260224" if run == "000001" else "260226"
		per_well = (
			output_root
			/ "Media_Density_T5"
			/ date
			/ "M08073"
			/ "AxonTracking"
			/ run
			/ "well000"
		)
		(per_well / "recon_outputs" / "synth_sorter_output").mkdir(parents=True)
		(per_well / "recon_outputs" / "cache" / "analyzers" / "segments").mkdir(parents=True)

	sessions = resolve_group_session_inputs(
		group_dataset_indices=[0, 1],
		well_id="well000",
		data_cfg=cfg,
		output_root=output_root,
	)
	assert len(sessions) == 2
	assert {s.dataset_index for s in sessions} == {0, 1}


def test_resolve_group_session_inputs_aggregates_errors(tmp_path: Path) -> None:
	# Both datasets are missing their synth folders → error message lists
	# both, not just the first.
	cfg = _data_cfg_with_datasets(
		_ds(date="260224", chip="M08073", well_ids=["well000"], run="000001"),
		_ds(date="260226", chip="M08073", well_ids=["well000"], run="000002"),
	)
	with pytest.raises(UnitmatchSessionInputMissing) as exc:
		resolve_group_session_inputs(
			group_dataset_indices=[0, 1],
			well_id="well000",
			data_cfg=cfg,
			output_root=tmp_path / "empty_root",
		)
	text = str(exc.value)
	assert "dataset=0" in text
	assert "dataset=1" in text


def test_resolve_group_session_inputs_out_of_range_index(tmp_path: Path) -> None:
	cfg = _data_cfg_with_datasets(_ds(date="260224", chip="M08073", well_ids=["well000"]))
	with pytest.raises(UnitmatchSessionInputMissing, match="out of range"):
		resolve_group_session_inputs(
			group_dataset_indices=[0, 99],
			well_id="well000",
			data_cfg=cfg,
			output_root=tmp_path / "nope",
			require_exists=True,
		)
