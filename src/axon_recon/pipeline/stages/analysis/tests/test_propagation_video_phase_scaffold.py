"""Slice 2 of analysis_propagation_video_plan: scaffold tests.

Verifies the phase wires into the analysis stage, defaults to disabled
(noop), and returns a `skipped: not_implemented_yet` marker when
enabled but the core impl (slice 4) hasn't shipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from axon_recon.pipeline.stages.analysis.config import (
	DEFAULT_ANALYSIS_PHASE_SEQUENCE,
	AnalysisStageConfig,
	parse_analysis_stage_config,
)
from axon_recon.pipeline.stages.analysis.orchestrators.propagation_video import (
	run_analysis_propagation_video,
)
from axon_recon.runtime_config import RuntimeConfig


_H5_TEMPLATE = "/pscratch/raw_data/proj/proj/{date}/{chip}/AxonTracking/{run}/data.raw.h5"


@dataclass
class _ShimStageConfig:
	parsed: AnalysisStageConfig
	propagation_video_enabled_override: bool | None = None

	def __getattr__(self, name: str) -> Any:
		if name == "propagation_video_enabled" and self.__dict__["propagation_video_enabled_override"] is not None:
			return self.__dict__["propagation_video_enabled_override"]
		return getattr(self.__dict__["parsed"], name)


def _parse_config(*, enabled: bool) -> _ShimStageConfig:
	payload = {
		"stages": {
			"analysis": {
				"phase_sequence": ["compute_metrics", "unitmatch", "propagation_video"],
				"phases": {
					"compute_metrics": {"enabled": True},
					"unitmatch": {"enabled": False},
					"propagation_video": {"enabled": enabled},
				},
			}
		}
	}
	parsed = parse_analysis_stage_config(runtime_config=RuntimeConfig(payload))
	return _ShimStageConfig(parsed=parsed, propagation_video_enabled_override=enabled)


# --- Slice 1 / 2 wiring tests ---------------------------------------------


def test_propagation_video_in_default_phase_sequence() -> None:
	assert "propagation_video" in DEFAULT_ANALYSIS_PHASE_SEQUENCE


def test_propagation_video_defaults_to_disabled() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig({"stages": {"analysis": {}}}),
	)
	assert cfg.propagation_video_enabled is False
	assert cfg.propagation_video_rel_output_root == "propagation_video"


def test_propagation_video_yaml_enable() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig(
			{
				"stages": {
					"analysis": {
						"phases": {"propagation_video": {"enabled": True}},
					}
				}
			}
		),
	)
	assert cfg.propagation_video_enabled is True


def test_yaml_render_knobs_default_to_v1_conventions() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig({"stages": {"analysis": {}}}),
	)
	# Slice 5: defaults match v1 conventions (PillowWriter ~20 FPS,
	# skip_frames=2 → halve the frame count, coolwarm cmap).
	assert cfg.propagation_video_fps == 20
	assert cfg.propagation_video_skip_frames == 2
	assert cfg.propagation_video_cmap == "coolwarm"


def test_yaml_render_knobs_can_be_overridden() -> None:
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig(
			{
				"stages": {
					"analysis": {
						"phases": {
							"propagation_video": {
								"enabled": True,
								"fps": 30,
								"skip_frames": 4,
								"cmap": "viridis",
							},
						}
					}
				}
			}
		),
	)
	assert cfg.propagation_video_fps == 30
	assert cfg.propagation_video_skip_frames == 4
	assert cfg.propagation_video_cmap == "viridis"


def test_yaml_alias_canonicalizes() -> None:
	# `video` and `prop_video` aliases should both resolve to the
	# canonical phase name in phase_sequence parsing.
	cfg = parse_analysis_stage_config(
		runtime_config=RuntimeConfig(
			{
				"stages": {
					"analysis": {
						"phase_sequence": ["compute_metrics", "video"],
					}
				}
			}
		),
	)
	assert "propagation_video" in cfg.phase_sequence


# --- Slice 2 orchestrator tests --------------------------------------------


def test_orchestrator_noop_when_disabled(tmp_path: Path) -> None:
	cfg = _parse_config(enabled=False)
	result = run_analysis_propagation_video(
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
	assert result["phase"] == "propagation_video"
	assert "disabled" in result["reason"]


def test_orchestrator_error_when_enabled_but_no_recon_outputs(tmp_path: Path) -> None:
	# Updated post-slice-7: orchestrator now does real unit discovery
	# when enabled. With no recon outputs on disk, the appropriate
	# response is `error: no unit_ids discovered` (not the slice-2
	# placeholder `skipped: not_implemented_yet`).
	cfg = _parse_config(enabled=True)
	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "error"
	assert "no unit_ids discovered" in result["reason"]


def test_orchestrator_writes_per_target_summary_on_disk(tmp_path: Path) -> None:
	cfg = _parse_config(enabled=False)
	run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=tmp_path / "outputs",
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	# Walk under tmp_path for the summary file.
	matches = list((tmp_path / "outputs").rglob("propagation_video_summary.json"))
	assert len(matches) == 1
	payload = json.loads(matches[0].read_text(encoding="utf-8"))
	assert payload["phase"] == "propagation_video"
	assert payload["status"] == "noop"


# --- Slice 7 orchestrator fan-out tests ---


def _scaffold_recon_outputs_with_units(
	tmp_path: Path,
	*,
	unit_ids: tuple[int, ...] = (5, 7),
) -> Path:
	"""Scaffold per-unit recon outputs that the inputs resolver can find."""

	output_root = tmp_path / "output_root"
	well_dir = (
		output_root
		/ "proj"
		/ "260224"
		/ "M08073"
		/ "AxonTracking"
		/ "000001"
		/ "well000"
	)
	for uid in unit_ids:
		unit_tmpl_dir = (
			well_dir / "recon_outputs" / "cache" / "templates" / "merged" / f"unit_{uid}"
		)
		unit_tmpl_dir.mkdir(parents=True, exist_ok=True)
		(unit_tmpl_dir / "merged_template.npy").write_bytes(b"\x00")
		(unit_tmpl_dir / "merged_channel_locations.npy").write_bytes(b"\x00")
		(unit_tmpl_dir / "unit_templates_summary.json").write_text("{}", encoding="utf-8")
		gtr_dir = well_dir / "recon_outputs" / "units" / f"{int(uid):04d}"
		gtr_dir.mkdir(parents=True, exist_ok=True)
		(gtr_dir / "gtr.pkl").write_bytes(b"\x00")
		(gtr_dir / "gtr.json").write_text("{}", encoding="utf-8")
	return output_root


def _mock_render(*, inputs, out_path, force_restart=False, **kwargs) -> dict:
	"""Mock that pretends to render and returns an ok payload. Writes a
	stub file so subsequent idempotency checks would find the output."""
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	Path(out_path).write_bytes(b"GIF89a-mock")
	return {
		"status": "ok",
		"reason": "rendered",
		"out_path": str(out_path),
		"unit_id": int(inputs.unit_id),
		"frames": 41,
		"cmap": "coolwarm",
		"fps": 20,
		"skip_frames": 2,
	}


def test_orchestrator_fans_out_across_discovered_units(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5, 7))
	cfg = _parse_config(enabled=True)
	# Inject the mock renderer via the test seam.
	cfg.__dict__["_propagation_video_render_override"] = _mock_render

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "ok"
	assert result["n_units_processed"] == 2
	assert result["n_units_ok"] == 2
	processed_unit_ids = sorted(u["unit_id"] for u in result["units_processed"])
	assert processed_unit_ids == [5, 7]
	# Per-unit GIFs landed at <stage>/propagation_video/unit_<NNNN>.gif.
	out_paths = sorted(Path(u["out_path"]).name for u in result["units_processed"])
	assert out_paths == ["unit_0005.gif", "unit_0007.gif"]


def test_orchestrator_error_when_no_units_discovered(tmp_path: Path) -> None:
	# No recon_outputs/cache/templates/merged/ → no units.
	output_root = tmp_path / "empty"
	cfg = _parse_config(enabled=True)
	cfg.__dict__["_propagation_video_render_override"] = _mock_render

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "error"
	assert "no unit_ids discovered" in result["reason"]


def test_orchestrator_records_missing_inputs_as_unit_level_error(tmp_path: Path) -> None:
	# Unit dir exists but gtr.pkl is missing → inputs resolver raises,
	# orchestrator records a unit-level "error" entry but DOESN'T crash
	# the whole target.
	output_root = tmp_path / "output_root"
	well_dir = output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000"
	unit_tmpl_dir = well_dir / "recon_outputs" / "cache" / "templates" / "merged" / "unit_5"
	unit_tmpl_dir.mkdir(parents=True, exist_ok=True)
	(unit_tmpl_dir / "merged_template.npy").write_bytes(b"\x00")
	(unit_tmpl_dir / "merged_channel_locations.npy").write_bytes(b"\x00")
	# DON'T create the GTR pickle.

	cfg = _parse_config(enabled=True)
	cfg.__dict__["_propagation_video_render_override"] = _mock_render

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "error"  # all units failed
	assert result["n_units_error"] == 1
	assert "missing inputs" in result["units_processed"][0]["reason"]


# --- Slice 6 dry-run tests ---


def test_dry_run_skips_render_and_writes_dry_run_ok(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5, 7))
	cfg = _parse_config(enabled=True)
	cfg.__dict__["dry_run"] = True

	# Render mock that flags if called (it shouldn't be).
	calls: list[Any] = []

	def _render_should_not_be_called(**kwargs):
		calls.append(kwargs)
		return {"status": "ok"}

	cfg.__dict__["_propagation_video_render_override"] = _render_should_not_be_called

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "dry_run_ok"
	assert result["reason"] == "dry_run: skipped renders"
	assert result["n_units_discovered"] == 2
	# Render was NOT called.
	assert calls == []
	# Inputs + outputs sections populated.
	assert len(result["inputs_resolved"]) == 6  # 2 units × 3 inputs
	assert len(result["outputs_would_produce"]) == 2  # one GIF per unit
	# All prerequisites exist (the scaffold wrote them).
	assert result["validation"]["missing_prerequisites"] == []


def test_dry_run_reports_missing_prerequisites_without_raising(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5,))
	# Remove the GTR pickle to simulate a missing prereq.
	gtr = output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000" / "recon_outputs" / "units" / "0005" / "gtr.pkl"
	gtr.unlink()

	cfg = _parse_config(enabled=True)
	cfg.__dict__["dry_run"] = True

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "dry_run_ok"
	# Missing prereq surfaced in validation.
	assert any(
		"gtr_pkl missing" in msg
		for msg in result["validation"]["missing_prerequisites"]
	)


def test_dry_run_summary_persists_to_disk(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5,))
	cfg = _parse_config(enabled=True)
	cfg.__dict__["dry_run"] = True

	run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	matches = list(output_root.rglob("propagation_video_summary.json"))
	assert len(matches) == 1
	payload = json.loads(matches[0].read_text(encoding="utf-8"))
	assert payload["status"] == "dry_run_ok"


def test_orchestrator_forwards_yaml_knobs_to_render(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5,))
	cfg = _parse_config(enabled=True)
	# Override YAML knobs on the shim. These should flow through to the
	# render callable.
	cfg.__dict__["propagation_video_fps"] = 30
	cfg.__dict__["propagation_video_skip_frames"] = 4
	cfg.__dict__["propagation_video_cmap"] = "viridis"

	render_kwargs_captured: list[dict] = []

	def _capture_render(*, inputs, out_path, force_restart=False, **kwargs):
		render_kwargs_captured.append(kwargs)
		Path(out_path).parent.mkdir(parents=True, exist_ok=True)
		Path(out_path).write_bytes(b"GIF")
		return {"status": "ok", "reason": "rendered", "out_path": str(out_path)}

	cfg.__dict__["_propagation_video_render_override"] = _capture_render

	run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert len(render_kwargs_captured) == 1
	assert render_kwargs_captured[0]["fps"] == 30
	assert render_kwargs_captured[0]["skip_frames"] == 4
	assert render_kwargs_captured[0]["cmap"] == "viridis"


def test_orchestrator_partial_status_when_some_units_fail(tmp_path: Path) -> None:
	output_root = _scaffold_recon_outputs_with_units(tmp_path, unit_ids=(5, 7))
	# Remove unit 7's gtr.pkl to simulate a partial failure.
	(output_root / "proj" / "260224" / "M08073" / "AxonTracking" / "000001" / "well000"
		/ "recon_outputs" / "units" / "0007" / "gtr.pkl").unlink()
	cfg = _parse_config(enabled=True)
	cfg.__dict__["_propagation_video_render_override"] = _mock_render

	result = run_analysis_propagation_video(
		dataset_index=0,
		dataset_id="ds0",
		h5_path=Path(_H5_TEMPLATE.format(date="260224", chip="M08073", run="000001")),
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="analysis_outputs",
		stage_config=cfg,
		force_restart=False,
	)
	assert result["status"] == "partial"
	assert result["n_units_ok"] == 1
	assert result["n_units_error"] == 1
