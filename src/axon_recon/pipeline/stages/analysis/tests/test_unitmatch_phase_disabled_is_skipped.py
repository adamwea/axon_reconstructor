"""Slice-1 + slice-3 tests for unitmatch_phase_plan: the phase wires in,
is a no-op when disabled, and invokes unitlink.match (mocked here) when
enabled with valid group inputs on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from axon_recon.pipeline.stages.analysis.config import (
	AnalysisStageConfig,
	parse_analysis_stage_config,
)
from axon_recon.pipeline.stages.analysis.orchestrators.unitmatch import (
	run_analysis_unitmatch,
)
from axon_recon.runtime_config import RuntimeConfig


_H5_TEMPLATE = "/pscratch/raw_data/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


@dataclass
class _ShimStageConfig:
	"""Wrapper that mimics AnalysisStageConfig attribute surface + adds
	test-only hooks (well_metadata_lookup override, unitlink.match mock)."""

	parsed: AnalysisStageConfig
	well_metadata_lookup: dict
	unitmatch_call_override: Any = None

	def __getattr__(self, name: str) -> Any:
		if name == "well_metadata_lookup":
			return self.__dict__["well_metadata_lookup"]
		if name == "_unitmatch_call_override":
			return self.__dict__["unitmatch_call_override"]
		return getattr(self.__dict__["parsed"], name)


def _make_stage_config(
	*,
	unitmatch_enabled: bool,
	well_metadata: dict | None = None,
	match_callable: Any = None,
) -> AnalysisStageConfig:
	payload = {
		"stages": {
			"analysis": {
				"phase_sequence": ["compute_metrics", "unitmatch"],
				"phases": {
					"compute_metrics": {"enabled": True},
					"unitmatch": {"enabled": unitmatch_enabled},
				},
			}
		}
	}
	parsed = parse_analysis_stage_config(runtime_config=RuntimeConfig(payload))
	if well_metadata is None and match_callable is None:
		return parsed
	return _ShimStageConfig(
		parsed=parsed,
		well_metadata_lookup=well_metadata or {},
		unitmatch_call_override=match_callable,
	)


# --- slice 1: scaffold + disabled wiring -------------------------------------


def test_parse_unitmatch_phase_block_defaults_disabled() -> None:
	cfg = _make_stage_config(unitmatch_enabled=False)
	assert cfg.unitmatch_enabled is False
	assert cfg.phase_sequence == ("compute_metrics", "unitmatch")


def test_parse_unitmatch_phase_block_can_be_enabled_in_yaml() -> None:
	cfg = _make_stage_config(unitmatch_enabled=True)
	assert cfg.unitmatch_enabled is True


def test_analysis_stage_default_sequence_includes_unitmatch() -> None:
	cfg = parse_analysis_stage_config(runtime_config=RuntimeConfig({"stages": {"analysis": {}}}))
	assert "unitmatch" in cfg.phase_sequence
	assert cfg.unitmatch_enabled is False


def test_run_analysis_unitmatch_returns_noop_when_disabled(tmp_path: Path) -> None:
	cfg = _make_stage_config(unitmatch_enabled=False)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "noop"
	assert result["phase"] == "unitmatch"
	assert "disabled" in result["reason"]


# --- slice 3: enabled path with mocked unitlink.match -------------------------


def _scaffold_well_outputs(
	tmp_path: Path,
	*,
	chip: str,
	well_id: str,
	dates_and_runs: list[tuple[str, str]],
) -> Path:
	"""Scaffold per-well recon_outputs trees expected by resolve_session_inputs."""

	output_root = tmp_path / "output_root"
	for date, run in dates_and_runs:
		per_well = (
			output_root
			/ "proj"
			/ date
			/ chip
			/ "AxonTracking"
			/ run
			/ well_id
		)
		(per_well / "recon_outputs" / "synth_sorter_output").mkdir(parents=True)
		(per_well / "recon_outputs" / "cache" / "analyzers" / "segments").mkdir(parents=True)
	return output_root


def _well_metadata_for_group(
	*,
	chip: str,
	well_id: str,
	dates_and_runs: list[tuple[str, str]],
) -> dict:
	out = {}
	for idx, (date, run) in enumerate(dates_and_runs):
		out[(idx, well_id)] = {
			"chip_id": chip,
			"raw_data_h5_path": _H5_TEMPLATE.format(date=date, chip=chip, run=run),
			"DIV": int(date[-2:]),
			"dataset_id": f"ds_{idx}",
		}
	return out


def test_unitmatch_orchestrator_runs_unitlink_match_once_per_group(tmp_path: Path) -> None:
	chip = "M08073"
	well = "well000"
	output_root = _scaffold_well_outputs(
		tmp_path, chip=chip, well_id=well, dates_and_runs=[("260224", "000001"), ("260226", "000002")]
	)
	well_metadata = _well_metadata_for_group(
		chip=chip, well_id=well, dates_and_runs=[("260224", "000001"), ("260226", "000002")]
	)
	calls: list[dict[str, Any]] = []

	class _StubResult:
		def __init__(self) -> None:
			self.match_table = [{"a": 1}, {"a": 2}, {"a": 3}]
			self.uid_assignment = [{"u": 0}, {"u": 1}]
			self.summary = {"backend": "classical"}

	def _stub_match(*, sorter_outputs, out_folder, match_threshold):
		calls.append(
			{
				"sorter_outputs": [str(p) for p in sorter_outputs],
				"out_folder": str(out_folder),
				"match_threshold": float(match_threshold),
			}
		)
		Path(out_folder).mkdir(parents=True, exist_ok=True)
		(Path(out_folder) / "summary.json").write_text(
			json.dumps({"backend": "classical", "n_sessions": len(sorter_outputs)}),
			encoding="utf-8",
		)
		return _StubResult()

	cfg = _make_stage_config(
		unitmatch_enabled=True, well_metadata=well_metadata, match_callable=_stub_match
	)

	result0 = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip=chip, run="000001")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result0["status"] == "ok"
	assert result0["chip_id"] == chip
	assert result0["group_dataset_indices"] == [0, 1]
	assert result0["match_table_rows"] == 3
	assert len(calls) == 1
	assert len(calls[0]["sorter_outputs"]) == 2

	# Second target in the same group → short-circuit.
	result1 = run_analysis_unitmatch(
		dataset_index=1,
		dataset_id="ds_1",
		h5_path=Path(_H5_TEMPLATE.format(date="260226", chip=chip, run="000002")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result1["status"] == "skipped"
	assert len(calls) == 1


def test_unitmatch_orchestrator_writes_outputs_under_chip_well(tmp_path: Path) -> None:
	chip = "M08073"
	well = "well000"
	output_root = _scaffold_well_outputs(
		tmp_path, chip=chip, well_id=well, dates_and_runs=[("260224", "000001")]
	)
	well_metadata = _well_metadata_for_group(
		chip=chip, well_id=well, dates_and_runs=[("260224", "000001")]
	)

	def _stub_match(*, sorter_outputs, out_folder, match_threshold):
		Path(out_folder).mkdir(parents=True, exist_ok=True)
		(Path(out_folder) / "match_table.tsv").write_text("x\n1\n", encoding="utf-8")
		(Path(out_folder) / "summary.json").write_text("{}", encoding="utf-8")

		class _R:
			match_table = [{"a": 1}]
			uid_assignment = [{"u": 0}]
			summary = {"backend": "classical"}

		return _R()

	cfg = _make_stage_config(
		unitmatch_enabled=True, well_metadata=well_metadata, match_callable=_stub_match
	)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip=chip, run="000001")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "ok"
	group_dir = Path(result["group_output_dir"])
	assert group_dir.name == well
	assert group_dir.parent.name == chip
	assert (group_dir / "match_table.tsv").is_file()


def test_unitmatch_orchestrator_force_restart_reruns(tmp_path: Path) -> None:
	chip = "M08073"
	well = "well000"
	output_root = _scaffold_well_outputs(
		tmp_path, chip=chip, well_id=well, dates_and_runs=[("260224", "000001")]
	)
	well_metadata = _well_metadata_for_group(
		chip=chip, well_id=well, dates_and_runs=[("260224", "000001")]
	)
	call_count = {"n": 0}

	def _stub_match(*, sorter_outputs, out_folder, match_threshold):
		call_count["n"] += 1
		Path(out_folder).mkdir(parents=True, exist_ok=True)
		(Path(out_folder) / "summary.json").write_text("{}", encoding="utf-8")

		class _R:
			match_table = []
			uid_assignment = []
			summary = {}

		return _R()

	cfg = _make_stage_config(
		unitmatch_enabled=True, well_metadata=well_metadata, match_callable=_stub_match
	)
	run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip=chip, run="000001")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip=chip, run="000001")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=True,
	)
	assert call_count["n"] == 2


def test_unitmatch_orchestrator_records_missing_session_inputs_as_error(tmp_path: Path) -> None:
	chip = "M08073"
	well = "well000"
	output_root = tmp_path / "output_root_empty"
	well_metadata = _well_metadata_for_group(
		chip=chip, well_id=well, dates_and_runs=[("260224", "000001")]
	)

	def _stub_match(**kwargs):
		raise AssertionError("unitlink.match should not be reached when inputs missing")

	cfg = _make_stage_config(
		unitmatch_enabled=True, well_metadata=well_metadata, match_callable=_stub_match
	)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip=chip, run="000001")),
		stream_id=well,
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "error"
	assert "missing session inputs" in result["reason"]


def test_unitmatch_orchestrator_records_unknown_chip_as_error(tmp_path: Path) -> None:
	cfg = _make_stage_config(
		unitmatch_enabled=True, well_metadata={}, match_callable=lambda **_: None
	)
	result = run_analysis_unitmatch(
		dataset_index=0,
		dataset_id="ds_0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "out",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "error"
	assert "chip_id" in result["reason"]
