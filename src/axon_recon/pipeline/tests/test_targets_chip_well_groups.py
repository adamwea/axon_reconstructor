"""Slice 4 of unitmatch_phase_plan.md: ``--targets chip-well:<chip>:<well>``
syntax is parsed by the CLI and expanded against the data config by
``select_execution_targets``.

Tests cover the two new layers:

1. CLI parsing (``_parse_targets_pairs_from_args`` ignores chip-well: tokens,
   and ``_parse_targets_chip_well_groups_from_args`` picks them up).
2. ``select_execution_targets`` merges chip-well groups into the per-pair
   allowlist by matching each group's chip_id against every dataset's
   ``raw_data_h5_path``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from axon_recon.pipeline.cli import (
	_parse_targets_chip_well_groups_from_args,
	_parse_targets_pairs_from_args,
)
from axon_recon.pipeline.config import (
	PipelineRuntimeBundle,
	get_target_pairs_override,
	select_execution_targets,
	set_target_chip_well_groups_override,
	set_target_pairs_override,
	set_target_wells_override,
)
from axon_recon.runtime_config import RuntimeConfig


# --- CLI parser tests ---------------------------------------------------------


def test_parse_pairs_ignores_chip_well_tokens_and_returns_only_pairs() -> None:
	args = SimpleNamespace(targets=["0:0,chip-well:M08073:well000"])
	pairs = _parse_targets_pairs_from_args(args)
	assert pairs == {0: ["well000"]}


def test_parse_pairs_returns_none_when_only_chip_well_tokens() -> None:
	args = SimpleNamespace(targets=["chip-well:M08073:well000"])
	assert _parse_targets_pairs_from_args(args) is None


def test_parse_chip_well_groups_extracts_chip_and_well() -> None:
	args = SimpleNamespace(
		targets=["chip-well:M08073:well000", "chip-well:M08074:1"]
	)
	groups = _parse_targets_chip_well_groups_from_args(args)
	assert groups == [("M08073", "well000"), ("M08074", "well001")]


def test_parse_chip_well_groups_returns_none_when_only_pair_tokens() -> None:
	args = SimpleNamespace(targets=["0:0,1:0"])
	assert _parse_targets_chip_well_groups_from_args(args) is None


def test_parse_chip_well_groups_rejects_malformed_token() -> None:
	args = SimpleNamespace(targets=["chip-well:M08073"])  # missing well
	with pytest.raises(SystemExit):
		_parse_targets_chip_well_groups_from_args(args)


def test_parse_chip_well_groups_rejects_blank_chip() -> None:
	args = SimpleNamespace(targets=["chip-well::well000"])
	with pytest.raises(SystemExit):
		_parse_targets_chip_well_groups_from_args(args)


def test_parse_pairs_rejects_token_with_no_colon() -> None:
	args = SimpleNamespace(targets=["nocolon"])
	with pytest.raises(SystemExit):
		_parse_targets_pairs_from_args(args)


# --- select_execution_targets integration tests -------------------------------


_H5_TEMPLATE = (
	"/raw/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"
)


def _make_dataset(
	*, date: str, chip: str, run: str, wells: list[str], include: bool = True
) -> dict:
	return {
		"include_in_runtime": bool(include),
		"raw_data_h5_path": _H5_TEMPLATE.format(date=date, chip=chip, run=run),
		"DIV": int(date[-2:]),
		"wells": [{"well_id": w, "include_in_runtime": True} for w in wells],
	}


def _build_bundle(tmp_path: Path, datasets: list[dict]) -> PipelineRuntimeBundle:
	data_config = RuntimeConfig(
		{
			"output_root": str(tmp_path / "outputs"),
			"datasets": datasets,
		}
	)
	runtime_config = RuntimeConfig({"stages": {}})
	return PipelineRuntimeBundle(
		runtime_config_path=tmp_path / "runtime.yml",
		data_config_path=tmp_path / "data.yml",
		runtime_config=runtime_config,
		data_config=data_config,
	)


@pytest.fixture(autouse=True)
def _reset_target_overrides():
	"""Ensure no override leaks between tests."""
	set_target_pairs_override(None)
	set_target_wells_override(None)
	set_target_chip_well_groups_override(None)
	yield
	set_target_pairs_override(None)
	set_target_wells_override(None)
	set_target_chip_well_groups_override(None)


def test_chip_well_group_expands_to_matching_datasets(tmp_path: Path) -> None:
	bundle = _build_bundle(
		tmp_path,
		datasets=[
			_make_dataset(date="260224", chip="M08073", run="000001", wells=["well000", "well001"]),
			_make_dataset(date="260226", chip="M08073", run="000002", wells=["well000", "well001"]),
			_make_dataset(date="260224", chip="M08074", run="000003", wells=["well000"]),
		],
	)
	set_target_chip_well_groups_override([("M08073", "well000")])

	# Build the (h5, well) files the loader needs to exist so the path resolver
	# doesn't bail. We don't actually need the files for target selection, but
	# select_execution_targets only inspects datasets and produces ExecutionTarget
	# objects — it doesn't touch the h5 files.
	targets = select_execution_targets(bundle=bundle)

	# Both M08073 datasets matched; only well000 of each.
	resolved = sorted((int(t.dataset_index), str(t.stream_id)) for t in targets)
	assert resolved == [(0, "well000"), (1, "well000")]

	# The override was merged into the per-pair allowlist.
	pair_override = get_target_pairs_override()
	assert pair_override == {0: {"well000"}, 1: {"well000"}}


def test_chip_well_group_merges_with_existing_pair_override(tmp_path: Path) -> None:
	bundle = _build_bundle(
		tmp_path,
		datasets=[
			_make_dataset(date="260224", chip="M08073", run="000001", wells=["well000"]),
			_make_dataset(date="260226", chip="M08073", run="000002", wells=["well000"]),
			_make_dataset(date="260224", chip="M08074", run="000003", wells=["well000"]),
		],
	)
	# Operator passed both forms: explicit pair AND a chip-well group.
	set_target_pairs_override({2: ["well000"]})
	set_target_chip_well_groups_override([("M08073", "well000")])

	targets = select_execution_targets(bundle=bundle)
	resolved = sorted((int(t.dataset_index), str(t.stream_id)) for t in targets)
	assert resolved == [(0, "well000"), (1, "well000"), (2, "well000")]


def test_chip_well_group_unknown_chip_matches_nothing(tmp_path: Path) -> None:
	bundle = _build_bundle(
		tmp_path,
		datasets=[
			_make_dataset(date="260224", chip="M08073", run="000001", wells=["well000"]),
		],
	)
	set_target_chip_well_groups_override([("UNKNOWN", "well000")])
	with pytest.raises(ValueError, match="No enabled datasets matched --targets chip-well groups"):
		select_execution_targets(bundle=bundle)
