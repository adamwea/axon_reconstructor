from __future__ import annotations

from pathlib import Path

import yaml

from axon_recon.pipeline.status import (
	STAGE_PHASES,
	STAGE_WELL_MARKER,
	_rel_pattern_from_h5,
	format_default_tables,
	format_verbose_tables,
	incomplete_dataset_indices,
	scan_status,
)


def _write_runtime_and_data(tmp_path: Path, datasets: list[dict]) -> Path:
	data_yml = tmp_path / "data.yml"
	data_payload = {
		"output_root": str(tmp_path / "out"),
		"datasets": datasets,
	}
	data_yml.write_text(yaml.safe_dump(data_payload))
	runtime_yml = tmp_path / "runtime.yml"
	runtime_yml.write_text(yaml.safe_dump({"data": "./data.yml"}))
	return runtime_yml


def _mkmarker(output_root: Path, rel_pattern: str, well_id: str, marker_parts: tuple[str, ...]) -> None:
	target = output_root / rel_pattern / well_id
	for part in marker_parts:
		target = target / part
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text("{}")


def _build_dataset(raw_h5: str, well_ids: list[str], *, div: int = 7) -> dict:
	return {
		"raw_data_h5_path": raw_h5,
		"DIV": div,
		"include_in_runtime": True,
		"wells": [{"well_id": wid, "include_in_runtime": True} for wid in well_ids],
	}


def test_scan_status_detects_per_stage_completion(tmp_path: Path) -> None:
	raw = "/data/proj/260224/M08073/AxonTracking/000031/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000", "well001"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)
	output_root = tmp_path / "out"

	# well000: preprocess+spikesort done. well001: only preprocess done.
	_mkmarker(output_root, rel_pattern, "well000", STAGE_WELL_MARKER["preprocess"])
	_mkmarker(output_root, rel_pattern, "well000", STAGE_WELL_MARKER["spikesort"])
	_mkmarker(output_root, rel_pattern, "well001", STAGE_WELL_MARKER["preprocess"])

	report = scan_status(runtime_yml)

	stage_by_name = {stage.stage: stage for stage in report.stages}
	preproc = stage_by_name["preprocess"].datasets[0]
	assert [(w.well_id, w.stage_done) for w in preproc.wells] == [("well000", True), ("well001", True)]
	spike = stage_by_name["spikesort"].datasets[0]
	assert [(w.well_id, w.stage_done) for w in spike.wells] == [("well000", True), ("well001", False)]
	recon = stage_by_name["reconstruct"].datasets[0]
	assert all(not w.stage_done for w in recon.wells)


def test_scan_status_respects_target_wells_filter(tmp_path: Path) -> None:
	raw = "/data/proj/260224/M08073/AxonTracking/000031/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000", "well001", "well002"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	report = scan_status(runtime_yml, target_wells=["well001"])

	# Every stage row in the single dataset should now show just one well.
	for stage_status in report.stages:
		ds = stage_status.datasets[0]
		well_ids = [w.well_id for w in ds.wells]
		assert well_ids == ["well001"], f"stage {stage_status.stage}: {well_ids}"


def test_scan_status_respects_target_datasets_filter(tmp_path: Path) -> None:
	datasets = [
		_build_dataset("/data/a/A/X/0001/data.raw.h5", ["well000"], div=5),
		_build_dataset("/data/b/B/Y/0002/data.raw.h5", ["well000"], div=10),
		_build_dataset("/data/c/C/Z/0003/data.raw.h5", ["well000"], div=15),
	]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	report = scan_status(runtime_yml, target_datasets=[0, 2])

	for stage_status in report.stages:
		seen_indices = [dataset.index for dataset in stage_status.datasets]
		assert seen_indices == [0, 2]


def test_scan_status_respects_stages_filter(tmp_path: Path) -> None:
	datasets = [_build_dataset("/d/p/M/X/0001/data.raw.h5", ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	report = scan_status(runtime_yml, stages=["spikesort", "analysis"])

	assert [stage.stage for stage in report.stages] == ["spikesort", "analysis"]


def test_scan_status_excludes_disabled_wells_and_datasets(tmp_path: Path) -> None:
	datasets = [
		{
			"raw_data_h5_path": "/d/p/M/X/0001/data.raw.h5",
			"DIV": 7,
			"include_in_runtime": True,
			"wells": [
				{"well_id": "well000", "include_in_runtime": True},
				{"well_id": "well999", "include_in_runtime": False},
			],
		},
		{
			"raw_data_h5_path": "/d/p/M/Y/0002/data.raw.h5",
			"DIV": 8,
			"include_in_runtime": False,
			"wells": [{"well_id": "well000", "include_in_runtime": True}],
		},
	]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	report = scan_status(runtime_yml)

	for stage_status in report.stages:
		assert [dataset.index for dataset in stage_status.datasets] == [0]
		dataset = stage_status.datasets[0]
		assert [well.well_id for well in dataset.wells] == ["well000"]


def test_scan_status_phase_markers_only_when_requested(tmp_path: Path) -> None:
	raw = "/d/p/M/X/0001/data.raw.h5"
	datasets = [_build_dataset(raw, ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)
	output_root = tmp_path / "out"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	bootstrap_marker = STAGE_PHASES["spikesort"][0][1]
	_mkmarker(output_root, rel_pattern, "well000", bootstrap_marker)

	default_report = scan_status(runtime_yml)
	spike = next(stage for stage in default_report.stages if stage.stage == "spikesort")
	assert spike.datasets[0].wells[0].phase_done == {}

	verbose_report = scan_status(runtime_yml, collect_phases=True)
	spike_v = next(stage for stage in verbose_report.stages if stage.stage == "spikesort")
	phases_for_well = spike_v.datasets[0].wells[0].phase_done
	assert phases_for_well["bootstrap_concat_binary"] is True
	assert phases_for_well["sort"] is False


def test_incomplete_dataset_indices(tmp_path: Path) -> None:
	raw_a = "/d/p/M/X/0001/data.raw.h5"
	raw_b = "/d/p/M/Y/0002/data.raw.h5"
	datasets = [
		_build_dataset(raw_a, ["well000", "well001"]),
		_build_dataset(raw_b, ["well000"]),
	]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)
	output_root = tmp_path / "out"
	_mkmarker(output_root, _rel_pattern_from_h5(Path(raw_a)), "well000", STAGE_WELL_MARKER["spikesort"])
	_mkmarker(output_root, _rel_pattern_from_h5(Path(raw_b)), "well000", STAGE_WELL_MARKER["spikesort"])

	report = scan_status(runtime_yml)
	assert incomplete_dataset_indices(report, stage="spikesort") == [0]


def test_format_default_tables_renders_each_stage(tmp_path: Path) -> None:
	datasets = [_build_dataset("/d/p/M/X/0001/data.raw.h5", ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	report = scan_status(runtime_yml)
	text = format_default_tables(report)

	for stage_name in ("preprocess", "spikesort", "reconstruct", "analysis"):
		assert f"=== {stage_name} ===" in text


def test_scan_status_records_acceptable_skip_from_well_marker(tmp_path: Path) -> None:
	from axon_recon.pipeline.status import ACCEPTABLE_SKIP_REASONS

	raw = "/d/p/M/X/0001/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	# Write merge_stage_summary.json with status=skipped, reason=no_qualifying_units.
	output_root = tmp_path / "out"
	marker_path = output_root / rel_pattern / "well000"
	for piece in STAGE_WELL_MARKER["spikesort"][:-1]:
		marker_path = marker_path / piece
	marker_path.mkdir(parents=True, exist_ok=True)
	(marker_path / STAGE_WELL_MARKER["spikesort"][-1]).write_text(
		'{"status": "skipped", "reason": "no_qualifying_units"}'
	)

	report = scan_status(runtime_yml, stages=["spikesort"])
	well = report.stages[0].datasets[0].wells[0]
	# stage_done is True because the marker file exists.
	assert well.stage_done is True
	# But there's a skip record on it that classifies as acceptable.
	assert well.has_skips is True
	assert well.has_unacceptable_skip is False
	skip = well.skip_records[0]
	assert skip.reason == "no_qualifying_units"
	assert skip.acceptable is True
	assert "no_qualifying_units" in ACCEPTABLE_SKIP_REASONS


def test_scan_status_flags_unacceptable_skip_from_well_marker(tmp_path: Path) -> None:
	raw = "/d/p/M/X/0001/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	output_root = tmp_path / "out"
	marker_path = output_root / rel_pattern / "well000"
	for piece in STAGE_WELL_MARKER["spikesort"][:-1]:
		marker_path = marker_path / piece
	marker_path.mkdir(parents=True, exist_ok=True)
	# Reason not in the allowlist → flagged as unacceptable.
	(marker_path / STAGE_WELL_MARKER["spikesort"][-1]).write_text(
		'{"status": "skipped", "reason": "manual_block_for_review"}'
	)

	report = scan_status(runtime_yml, stages=["spikesort"])
	well = report.stages[0].datasets[0].wells[0]
	assert well.has_skips is True
	assert well.has_unacceptable_skip is True
	assert well.skip_records[0].acceptable is False


def test_scan_status_no_skips_when_marker_status_is_ok(tmp_path: Path) -> None:
	raw = "/d/p/M/X/0001/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	output_root = tmp_path / "out"
	marker_path = output_root / rel_pattern / "well000"
	for piece in STAGE_WELL_MARKER["spikesort"][:-1]:
		marker_path = marker_path / piece
	marker_path.mkdir(parents=True, exist_ok=True)
	# Normal "ok" stage completion — no skip record should appear.
	(marker_path / STAGE_WELL_MARKER["spikesort"][-1]).write_text('{"status": "ok"}')

	report = scan_status(runtime_yml, stages=["spikesort"])
	well = report.stages[0].datasets[0].wells[0]
	assert well.stage_done is True
	assert well.has_skips is False


def test_format_default_tables_surfaces_skip_reasons_with_acceptability_tag(tmp_path: Path) -> None:
	raw = "/d/p/M/X/0001/data.raw.h5"
	rel_pattern = _rel_pattern_from_h5(Path(raw))
	datasets = [_build_dataset(raw, ["well000", "well001"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)

	output_root = tmp_path / "out"
	# well000 → acceptable skip (no_qualifying_units)
	# well001 → unacceptable skip (custom reason not in allowlist)
	for well_id, reason in (("well000", "no_qualifying_units"), ("well001", "weird_failure")):
		marker_path = output_root / rel_pattern / well_id
		for piece in STAGE_WELL_MARKER["spikesort"][:-1]:
			marker_path = marker_path / piece
		marker_path.mkdir(parents=True, exist_ok=True)
		(marker_path / STAGE_WELL_MARKER["spikesort"][-1]).write_text(
			f'{{"status": "skipped", "reason": "{reason}"}}'
		)

	report = scan_status(runtime_yml, stages=["spikesort"])
	text = format_default_tables(report)

	# Acceptable skip is tagged (ok).
	assert "well000[spikesort=no_qualifying_units(ok)]" in text
	# Unacceptable skip is tagged (!!).
	assert "well001[spikesort=weird_failure(!!)]" in text
	# Summary line mentions the skip counts.
	assert "2 skipped-but-complete wells (1 flagged !!)" in text


def test_format_verbose_tables_numbers_phase_legend(tmp_path: Path) -> None:
	raw = "/d/p/M/X/0001/data.raw.h5"
	datasets = [_build_dataset(raw, ["well000"])]
	runtime_yml = _write_runtime_and_data(tmp_path, datasets)
	output_root = tmp_path / "out"
	# Mark the first spikesort phase so we can see the glyph reflect it
	_mkmarker(output_root, _rel_pattern_from_h5(Path(raw)), "well000", STAGE_PHASES["spikesort"][0][1])

	report = scan_status(runtime_yml, stages=["spikesort"], collect_phases=True)
	text = format_verbose_tables(report)

	assert "=== spikesort phases ===" in text
	assert "1. bootstrap_concat_binary" in text
	assert "2. sort" in text
	# The glyph string should start with ✓ (phase 1 marker present) followed by ·'s
	assert "✓·······" in text
