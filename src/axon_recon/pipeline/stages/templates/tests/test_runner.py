from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.templates.models.inputs import (
	DataQualityChecksOutputsConfig,
	FootprintGridsReportConfig,
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	FootprintPlotsConfig,
	MultipleNegativePeaksCheckConfig,
	MultipleNegativePeaksOutputsConfig,
	MultiSourcePdfReportConfig,
	PerUnitQualityChecksOutputsConfig,
	PerUnitTemplatesOutputsConfig,
	ProbeGeometryConfig,
	PropagationPlotConfig,
	QualityCheckJsonOutputConfig,
	QualityCheckPlotOutputConfig,
	QualityChecksConfig,
	ReportsConfig,
	TemplateArtifactConfig,
	TemplateCirclesPlotConfig,
	TemplateWaveformOverlayConfig,
	TemplatePlotConfig,
	TimeUpsampleConfig,
	TemplatesInputs,
	TopographicalFootprintConfig,
	TopographicalFootprintsConfig,
	WfOverlayGridReportConfig,
)
from axon_recon.pipeline.stages.templates.runner import run_templates_stage


def _make_templates_artifacts(well_out_dir: Path) -> None:
	merged_unit_dir = well_out_dir / "templates_outputs" / "templates" / "merged" / "unit_94"
	full_unit_dir = well_out_dir / "templates_outputs" / "templates" / "full" / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)
	full_unit_dir.mkdir(parents=True, exist_ok=True)

	t = np.linspace(-1.0, 1.0, 40)
	merged_template = np.vstack(
		[
			np.sin(3.0 * t),
			np.sin(5.0 * t) * 0.6,
			np.sin(7.0 * t) * 0.3,
		]
	)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float)

	np.save(merged_unit_dir / "merged_contributing_template.npy", merged_template)
	np.save(merged_unit_dir / "merged_contributing_channel_locations.npy", merged_locs)

	full_template = np.zeros((6, 40), dtype=float)
	full_template[0:3, :] = merged_template
	full_locs = np.asarray(
		[
			[0.0, 0.0],
			[20.0, 0.0],
			[10.0, 18.0],
			[30.0, 10.0],
			[35.0, 20.0],
			[40.0, 25.0],
		],
		dtype=float,
	)
	np.save(full_unit_dir / "full_template.npy", full_template)
	np.save(full_unit_dir / "full_channel_locations_xy.npy", full_locs)

	# Materialized per-unit overlay payload for top-channel waveform plotting.
	wf = np.tile(np.sin(np.linspace(-1.0, 1.0, 40, dtype=float)), (120, 1))
	np.save(merged_unit_dir / "overlay_top_channel_waveforms.npy", wf)
	(merged_unit_dir / "overlay_top_channel_meta.json").write_text(
		json.dumps({"top_electrode_id": 0, "total_waveforms_at_channel": int(wf.shape[0])}),
		encoding="utf-8",
	)


def test_run_templates_stage_writes_png(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(
				write_png=True,
				write_svg=False,
				relpath="template",
				channel_scope="recorded_channels",
			),
		),
		unit_ids=[94],
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	unit_png = well_out_dir / "templates_outputs" / "units" / "0094" / "template.png"
	merged_npy = well_out_dir / "templates_outputs" / "units" / "0094" / "merged_template.npy"
	assert unit_png.exists()
	assert merged_npy.exists()
	assert str(unit_png) == result.units[0].outputs.get("template_png")
	assert str(merged_npy) == result.units[0].outputs.get("merged_template_npy")


def test_run_templates_stage_writes_template_circles_png(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=True, write_svg=False, relpath="template_circles"),
		),
		unit_ids=[94],
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	circles_png = well_out_dir / "templates_outputs" / "units" / "0094" / "template_circles.png"
	assert circles_png.exists()
	assert str(circles_png) == result.units[0].outputs.get("template_circles_png")


def test_run_templates_stage_writes_multiple_negative_peaks_quality_artifacts(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	merged_unit_dir = well_out_dir / "templates_outputs" / "templates" / "merged" / "unit_94"
	full_unit_dir = well_out_dir / "templates_outputs" / "templates" / "full" / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)
	full_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template = np.asarray(
		[
			[0.0, -0.8, -5.0, -1.2, -4.2, -0.7, 0.0],
			[0.0, -0.3, -1.1, -0.2, -0.2, -0.1, 0.0],
		],
		dtype=float,
	)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float)
	full_template = merged_template.copy()
	full_locs = merged_locs.copy()
	merged_eids = np.asarray([111, 222], dtype=int)

	np.save(merged_unit_dir / "merged_contributing_template.npy", merged_template)
	np.save(merged_unit_dir / "merged_contributing_channel_locations.npy", merged_locs)
	np.save(merged_unit_dir / "merged_contributing_electrode_ids.npy", merged_eids)
	np.save(full_unit_dir / "full_template.npy", full_template)
	np.save(full_unit_dir / "full_channel_locations_xy.npy", full_locs)

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", lambda **kwargs: {})

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
			footprint_plots=FootprintPlotsConfig(
				amplitude_map=FootprintMapConfig(write_png=False, write_svg=False),
				latency_map=FootprintMapConfig(write_png=False, write_svg=False),
			),
			topographical_footprints=TopographicalFootprintsConfig(
				amplitude=TopographicalFootprintConfig(write_png=False, write_svg=False),
				latency=TopographicalFootprintConfig(write_png=False, write_svg=False),
			),
			propagation_plots=PropagationPlotConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False)),
		quality_checks=QualityChecksConfig(
			enable=True,
			check_for_multiple_peaks_at_channel_templates=MultipleNegativePeaksCheckConfig(
				enable=True,
				prominence_fraction=0.30,
				min_separation_samples=2,
			),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	unit_summary_json = well_out_dir / "templates_outputs" / "units" / "0094" / "unit_templates_summary.json"
	unit_payload = json.loads(unit_summary_json.read_text(encoding="utf-8"))
	qc_payload = unit_payload["quality_checks"]["check_for_multiple_peaks_at_channel_templates"]
	assert qc_payload["detected"] is True
	assert qc_payload["violation_count"] == 1

	qc_json = well_out_dir / "templates_outputs" / "units" / "0094" / "quality_checks_multiple_negative_peaks.json"
	assert qc_json.exists()

	agg_json = well_out_dir / "templates_outputs" / "quality_checks_multiple_negative_peaks.json"
	agg_payload = json.loads(agg_json.read_text(encoding="utf-8"))
	assert agg_payload["units_with_violations"] == 1
	assert agg_payload["total_violations"] == 1


def test_run_templates_stage_quality_check_violation_plot_and_output_knobs(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	merged_unit_dir = well_out_dir / "templates_outputs" / "templates" / "merged" / "unit_94"
	full_unit_dir = well_out_dir / "templates_outputs" / "templates" / "full" / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)
	full_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template = np.asarray(
		[
			[0.0, -1.0, -6.0, -1.0, -4.0, -0.5, 0.0],
			[0.0, -0.9, -5.0, -1.0, -3.2, -0.4, 0.0],
		],
		dtype=float,
	)
	merged_locs = np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	full_template = merged_template.copy()
	full_locs = merged_locs.copy()

	np.save(merged_unit_dir / "merged_contributing_template.npy", merged_template)
	np.save(merged_unit_dir / "merged_contributing_channel_locations.npy", merged_locs)
	np.save(full_unit_dir / "full_template.npy", full_template)
	np.save(full_unit_dir / "full_channel_locations_xy.npy", full_locs)

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", lambda **kwargs: {})

	prop_calls: list[dict[str, Any]] = []

	def _fake_propagation(**kwargs):
		prop_calls.append(dict(kwargs))
		out: dict[str, str] = {}
		if bool(kwargs["config"].write_png):
			out["propagation_plot_png"] = str(kwargs["png_path"])
		if bool(kwargs.get("write_svg", False)):
			out["propagation_plot_svg"] = str(kwargs["svg_path"])
		return out

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", _fake_propagation)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			quality_checks=PerUnitQualityChecksOutputsConfig(
				check_for_multiple_peaks_at_channel_templates=MultipleNegativePeaksOutputsConfig(
					write_json=False,
					json_relpath="qc/unit_quality.json",
					plot=QualityCheckPlotOutputConfig(
						write_png=True,
						write_svg=True,
						relpath="qc/violating_channels",
						show_multiple_peak_markers=True,
						delay_peak_marker_color="red",
					),
				),
			),
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
			footprint_plots=FootprintPlotsConfig(
				amplitude_map=FootprintMapConfig(write_png=False, write_svg=False),
				latency_map=FootprintMapConfig(write_png=False, write_svg=False),
			),
			topographical_footprints=TopographicalFootprintsConfig(
				amplitude=TopographicalFootprintConfig(write_png=False, write_svg=False),
				latency=TopographicalFootprintConfig(write_png=False, write_svg=False),
			),
			propagation_plots=PropagationPlotConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False)),
		quality_checks=QualityChecksConfig(
			enable=True,
			check_for_multiple_peaks_at_channel_templates=MultipleNegativePeaksCheckConfig(
				enable=True,
				prominence_fraction=0.30,
				min_separation_samples=2,
			),
		),
		quality_checks_outputs=DataQualityChecksOutputsConfig(
			check_for_multiple_peaks_at_channel_templates=QualityCheckJsonOutputConfig(
				write_json=False,
				json_relpath="qc/run_level_quality.json",
			),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	unit_outputs = result.units[0].outputs
	assert "quality_checks_multiple_negative_peaks_json" not in unit_outputs
	assert unit_outputs["quality_checks_multiple_negative_peaks_plot_png"].endswith("units/0094/qc/violating_channels.png")
	assert unit_outputs["quality_checks_multiple_negative_peaks_plot_svg"].endswith("units/0094/qc/violating_channels.svg")

	qc_calls = [c for c in prop_calls if c.get("channel_indices") is not None]
	assert len(qc_calls) == 1
	assert sorted(list(qc_calls[0]["channel_indices"])) == [0, 1]
	assert qc_calls[0]["config"].show_multiple_peak_markers is True
	assert qc_calls[0]["config"].delay_peak_marker_color == "red"
	assert qc_calls[0]["peak_indices_by_channel"] == {0: [2, 4], 1: [2, 4]}

	assert not (well_out_dir / "templates_outputs" / "qc" / "run_level_quality.json").exists()


def test_run_templates_stage_propagation_right_panel_composes_svg(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", lambda **kwargs: {})
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", lambda **kwargs: {})

	circles_calls: list[dict[str, Any]] = []

	def _fake_circles(**kwargs):
		circles_calls.append(dict(kwargs))
		return {
			"template_circles_svg": str(kwargs["svg_path"]),
			"template_circles_png": str(kwargs["png_path"]),
		}

	prop_calls: list[dict[str, Any]] = []

	def _fake_prop(**kwargs):
		prop_calls.append(dict(kwargs))
		out = {"propagation_plot_png": str(kwargs["png_path"])}
		if bool(kwargs.get("write_svg", False)):
			out["propagation_plot_svg"] = str(kwargs["svg_path"])
		return out

	compose_calls: list[dict[str, Any]] = []
	compose_png_calls: list[dict[str, Any]] = []

	def _fake_compose(**kwargs):
		compose_calls.append(dict(kwargs))
		Path(str(kwargs["output_svg_path"])).parent.mkdir(parents=True, exist_ok=True)
		Path(str(kwargs["output_svg_path"])).write_text("<svg/>", encoding="utf-8")
		return kwargs["output_svg_path"]

	def _fake_compose_png(**kwargs):
		compose_png_calls.append(dict(kwargs))
		Path(str(kwargs["output_png_path"])).parent.mkdir(parents=True, exist_ok=True)
		Path(str(kwargs["output_png_path"])).write_bytes(b"\x89PNG\r\n\x1a\n")
		return kwargs["output_png_path"]

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", _fake_circles)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", _fake_prop)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.compose_svg_side_by_side", _fake_compose)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.compose_png_side_by_side", _fake_compose_png)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
			footprint_plots=FootprintPlotsConfig(
				amplitude_map=FootprintMapConfig(write_png=False, write_svg=False),
				latency_map=FootprintMapConfig(write_png=False, write_svg=False),
			),
			topographical_footprints=TopographicalFootprintsConfig(
				amplitude=TopographicalFootprintConfig(write_png=False, write_svg=False),
				latency=TopographicalFootprintConfig(write_png=False, write_svg=False),
			),
			propagation_plots=PropagationPlotConfig(
				write_pdf=False,
				write_png=True,
				show_right_panel=True,
				right_panel_svg_relpath="custom/right_panel.svg",
				right_panel_png_relpath="custom/right_panel.png",
				left_panel_png_dpi=550.0,
				right_panel_png_dpi=600.0,
				composed_png_dpi=700.0,
				right_panel_keep_temp_svg=False,
			),
		),
		reports=ReportsConfig(wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False)),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"
	assert len(compose_calls) == 1
	assert len(compose_png_calls) == 1
	assert len(circles_calls) >= 1
	right_panel_calls = [c for c in circles_calls if c.get("propagation_order_rank_by_channel") is not None]
	assert len(right_panel_calls) == 1
	right_panel_cfg = right_panel_calls[0]["config"]
	assert right_panel_cfg.dpi == 600.0
	assert str(right_panel_calls[0]["png_path"]).endswith("units/0094/custom/right_panel.png")
	assert str(right_panel_calls[0]["svg_path"]).endswith("units/0094/custom/right_panel.svg")
	assert str(compose_png_calls[0]["right_png_path"]).endswith("units/0094/custom/right_panel.png")
	assert compose_png_calls[0]["output_dpi"] == 700.0
	left_panel_calls = [c for c in prop_calls if bool(c.get("write_svg", False))]
	assert len(left_panel_calls) == 1
	assert left_panel_calls[0]["config"].left_panel_png_dpi == 550.0
	assert str(compose_calls[0]["right_svg_path"]).endswith("units/0094/custom/right_panel.svg")
	assert result.units[0].outputs["propagation_plot_svg"].endswith("units/0094/propagation_plot.svg")
	assert result.units[0].outputs["propagation_plot_png"].endswith("units/0094/propagation_plot.png")


def test_run_templates_stage_writes_channel_locations_for_all_template_artifacts(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			merged_template=TemplateArtifactConfig(
				write_npy=True,
				npy_relpath="arrays/merged.npy",
				channel_locations_npy_relpath="arrays/merged_locs.npy",
			),
			square_template=TemplateArtifactConfig(
				write_npy=True,
				npy_relpath="arrays/square.npy",
				channel_locations_npy_relpath="arrays/square_locs.npy",
				padding_value="zero",
			),
			scan_template=TemplateArtifactConfig(
				write_npy=True,
				npy_relpath="arrays/scan.npy",
				channel_locations_npy_relpath="arrays/scan_locs.npy",
				padding_value="zero",
			),
			full_template=TemplateArtifactConfig(
				write_npy=True,
				npy_relpath="arrays/full.npy",
				channel_locations_npy_relpath="arrays/full_locs.npy",
				padding_value="zero",
			),
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	unit_dir = well_out_dir / "templates_outputs" / "units" / "0094" / "arrays"
	merged_locs = np.load(unit_dir / "merged_locs.npy")
	square_locs = np.load(unit_dir / "square_locs.npy")
	scan_locs = np.load(unit_dir / "scan_locs.npy")
	full_locs = np.load(unit_dir / "full_locs.npy")

	np.testing.assert_allclose(
		merged_locs,
		np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float),
	)
	assert square_locs.shape == (4, 2)
	np.testing.assert_allclose(square_locs[:3, :], merged_locs)
	assert np.all(np.isnan(square_locs[3, :]))
	np.testing.assert_allclose(scan_locs, full_locs)

	assert result.units[0].outputs.get("merged_template_channel_locations_npy", "").endswith("arrays/merged_locs.npy")
	assert result.units[0].outputs.get("square_template_channel_locations_npy", "").endswith("arrays/square_locs.npy")
	assert result.units[0].outputs.get("scan_template_channel_locations_npy", "").endswith("arrays/scan_locs.npy")
	assert result.units[0].outputs.get("full_template_channel_locations_npy", "").endswith("arrays/full_locs.npy")


def test_run_templates_stage_writes_overlay_and_grid(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(
				write_pdf=False,
				write_png=True,
				png_relpath="template_wf_overlay.png",
				top_channels_per_template=2,
			),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(
				enabled=True,
				pdf_relpath="reports/template_multi_source.pdf",
			),
			replot_from_disk=True,
			time_upsample=TimeUpsampleConfig(enabled=True, factor=2, method="linear"),
			wf_overlay_grid=WfOverlayGridReportConfig(
				write_pdf=False,
				write_png=True,
				png_relpath="reports/wf_overlay_grid.png",
			)
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	overlay_png = well_out_dir / "templates_outputs" / "units" / "0094" / "template_wf_overlay.png"
	assert overlay_png.exists()
	assert str(overlay_png) == result.units[0].outputs.get("template_wf_overlay_png")

	grid_png = well_out_dir / "templates_outputs" / "reports" / "wf_overlay_grid.png"
	multi_source_pdf = well_out_dir / "templates_outputs" / "reports" / "template_multi_source.pdf"
	assert grid_png.exists()
	assert multi_source_pdf.exists()
	assert str(grid_png) == result.report_outputs.get("wf_overlay_grid_png")
	assert str(multi_source_pdf) == result.report_outputs.get("multi_source_pdf")

	summary_payload = (well_out_dir / "templates_outputs" / "templates_summary.json").read_text(encoding="utf-8")
	assert '"reports_replot_from_disk": true' in summary_payload
	assert '"factor": 2' in summary_payload


def test_run_templates_stage_writes_footprint_maps(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
			footprint_plots=FootprintPlotsConfig(
				amplitude_map=FootprintMapConfig(write_png=True, write_svg=False, relpath="maps/footprint_amp"),
				latency_map=FootprintMapConfig(write_png=True, write_svg=False, relpath="maps/footprint_lat"),
			),
			topographical_footprints=TopographicalFootprintsConfig(
				amplitude=TopographicalFootprintConfig(write_png=True, write_svg=False, relpath="maps/topo_amp"),
				latency=TopographicalFootprintConfig(write_png=True, write_svg=False, relpath="maps/topo_lat"),
			),
			propagation_plots=PropagationPlotConfig(
				write_pdf=True,
				pdf_relpath="maps/propagation.pdf",
				write_png=True,
				png_relpath="maps/propagation.png",
			),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
			footprint_grids=FootprintGridsReportConfig(
				circles_map_grid=FootprintMapGridReportConfig(
					write_pdf=False,
					write_png=True,
					png_relpath="reports/circles_map_grid.png",
				),
				amplitude_map_grid=FootprintMapGridReportConfig(
					write_pdf=False,
					write_png=True,
					png_relpath="reports/amplitude_map_grid.png",
				),
				latency_map_grid=FootprintMapGridReportConfig(
					write_pdf=False,
					write_png=True,
					png_relpath="reports/latency_map_grid.png",
				),
			),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	amp_png = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "footprint_amp.png"
	lat_png = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "footprint_lat.png"
	topo_amp_png = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "topo_amp.png"
	topo_lat_png = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "topo_lat.png"
	prop_png = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "propagation.png"
	prop_pdf = well_out_dir / "templates_outputs" / "units" / "0094" / "maps" / "propagation.pdf"
	assert amp_png.exists()
	assert lat_png.exists()
	assert topo_amp_png.exists()
	assert topo_lat_png.exists()
	assert prop_png.exists()
	assert prop_pdf.exists()
	assert str(amp_png) == result.units[0].outputs.get("footprint_amplitude_map_png")
	assert str(lat_png) == result.units[0].outputs.get("footprint_latency_map_png")
	assert str(topo_amp_png) == result.units[0].outputs.get("topographical_amplitude_footprint_png")
	assert str(topo_lat_png) == result.units[0].outputs.get("topographical_latency_footprint_png")
	assert str(prop_png) == result.units[0].outputs.get("propagation_plot_png")
	assert str(prop_pdf) == result.units[0].outputs.get("propagation_plot_pdf")

	amp_grid_png = well_out_dir / "templates_outputs" / "reports" / "amplitude_map_grid.png"
	circles_grid_png = well_out_dir / "templates_outputs" / "reports" / "circles_map_grid.png"
	lat_grid_png = well_out_dir / "templates_outputs" / "reports" / "latency_map_grid.png"
	assert circles_grid_png.exists()
	assert amp_grid_png.exists()
	assert lat_grid_png.exists()
	assert str(circles_grid_png) == result.report_outputs.get("template_circles_map_grid_png")
	assert str(amp_grid_png) == result.report_outputs.get("footprint_amplitude_map_grid_png")
	assert str(lat_grid_png) == result.report_outputs.get("footprint_latency_map_grid_png")


def test_run_templates_stage_reports_replot_from_disk_uses_unit_summaries(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	first_inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=True),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(enabled=False),
			replot_from_disk=False,
			time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="linear"),
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=True),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)
	first_result = run_templates_stage(first_inputs)
	assert len(first_result.units) == 1
	assert first_result.units[0].status == "ok"

	# Remove source templates artifacts to prove report replot can proceed from saved per-unit summaries only.
	shutil.rmtree(well_out_dir / "templates_outputs" / "templates" / "merged")

	second_inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=True),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(enabled=True, pdf_relpath="reports/template_multi_source.pdf"),
			replot_from_disk=True,
			time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="linear"),
			wf_overlay_grid=WfOverlayGridReportConfig(
				write_pdf=False,
				write_png=True,
				png_relpath="reports/wf_overlay_grid_replot.png",
			),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		n_jobs=1,
	)
	second_result = run_templates_stage(second_inputs)
	assert len(second_result.units) == 1
	assert second_result.units[0].status == "ok"

	replot_grid_png = well_out_dir / "templates_outputs" / "reports" / "wf_overlay_grid_replot.png"
	replot_multi_pdf = well_out_dir / "templates_outputs" / "reports" / "template_multi_source.pdf"
	assert replot_grid_png.exists()
	assert replot_multi_pdf.exists()
	assert str(replot_grid_png) == second_result.report_outputs.get("wf_overlay_grid_png")
	assert str(replot_multi_pdf) == second_result.report_outputs.get("multi_source_pdf")


def test_run_templates_stage_time_upsample_nearest_method(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=True),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(enabled=False),
			replot_from_disk=False,
			time_upsample=TimeUpsampleConfig(enabled=True, factor=3, method="nearest"),
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"
	assert "template_wf_overlay_png" in result.units[0].outputs


def test_run_templates_stage_uses_spikeinterface_materialization_fallback(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	called = {"value": False}

	def _fake_materialize(*, well_out_dir: Path, templates_out_dir: Path, unit_ids, include_concat: bool, include_segments: bool, **kwargs):
		called["value"] = True
		assert include_concat is True
		assert include_segments is True
		assert unit_ids == [94]
		assert "merge_method" in kwargs
		assert templates_out_dir == (well_out_dir / "templates_outputs")

		merged_dir = templates_out_dir / "templates" / "merged" / "unit_94"
		full_dir = templates_out_dir / "templates" / "full" / "unit_94"
		merged_dir.mkdir(parents=True, exist_ok=True)
		full_dir.mkdir(parents=True, exist_ok=True)

		t = np.vstack([np.sin(np.linspace(-1.0, 1.0, 40)), np.cos(np.linspace(-1.0, 1.0, 40))])
		locs = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float)
		np.save(merged_dir / "merged_contributing_template.npy", t)
		np.save(merged_dir / "merged_contributing_channel_locations.npy", locs)
		np.save(full_dir / "full_template.npy", t)
		np.save(full_dir / "full_channel_locations_xy.npy", locs)

		return merged_dir.parent, full_dir.parent

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.runner.materialize_templates_from_spikeinterface",
		_fake_materialize,
	)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert called["value"] is True
	assert len(result.units) == 1
	assert result.units[0].status == "ok"


def test_run_templates_stage_force_restart_prefers_spikeinterface_materialization(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)
	called = {"value": False}

	def _fake_materialize(*, well_out_dir: Path, templates_out_dir: Path, unit_ids, include_concat: bool, include_segments: bool, **kwargs):
		called["value"] = True
		assert include_concat is True
		assert include_segments is True
		assert unit_ids == [94]
		assert "merge_method" in kwargs
		assert templates_out_dir == (well_out_dir / "templates_outputs")

		merged_dir = templates_out_dir / "templates" / "merged" / "unit_94"
		full_dir = templates_out_dir / "templates" / "full" / "unit_94"
		merged_dir.mkdir(parents=True, exist_ok=True)
		full_dir.mkdir(parents=True, exist_ok=True)

		t = np.vstack([np.sin(np.linspace(-1.0, 1.0, 40)), np.cos(np.linspace(-1.0, 1.0, 40))])
		locs = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float)
		np.save(merged_dir / "merged_contributing_template.npy", t)
		np.save(merged_dir / "merged_contributing_channel_locations.npy", locs)
		np.save(full_dir / "full_template.npy", t)
		np.save(full_dir / "full_channel_locations_xy.npy", locs)

		return merged_dir.parent, full_dir.parent

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.runner.materialize_templates_from_spikeinterface",
		_fake_materialize,
	)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert called["value"] is True
	assert len(result.units) == 1
	assert result.units[0].status == "ok"


def test_run_templates_stage_writes_upsampling_decisions_to_summaries(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")

	def _fake_materialize(*, well_out_dir: Path, templates_out_dir: Path, unit_ids, include_concat: bool, include_segments: bool, **kwargs):
		assert include_concat is True
		assert include_segments is True
		assert unit_ids == [94]
		merged_dir = templates_out_dir / "templates" / "merged" / "unit_94"
		full_dir = templates_out_dir / "templates" / "full" / "unit_94"
		merged_dir.mkdir(parents=True, exist_ok=True)
		full_dir.mkdir(parents=True, exist_ok=True)
		t = np.vstack([np.sin(np.linspace(-1.0, 1.0, 40)), np.cos(np.linspace(-1.0, 1.0, 40))])
		locs = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float)
		np.save(merged_dir / "merged_contributing_template.npy", t)
		np.save(merged_dir / "merged_contributing_channel_locations.npy", locs)
		np.save(full_dir / "full_template.npy", t)
		np.save(full_dir / "full_channel_locations_xy.npy", locs)
		return (
			merged_dir.parent,
			full_dir.parent,
			{
				94: {
					"enabled": True,
					"applied": True,
					"factor": 10,
					"method": "sinc",
					"raw_hz": 10000.0,
					"analyzer_hz": 10000.0,
					"target_hz": 100000.0,
					"skip_reason": None,
				}
			},
		)

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.runner.materialize_templates_from_spikeinterface",
		_fake_materialize,
	)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	unit_summary_json = well_out_dir / "templates_outputs" / "units" / "0094" / "unit_templates_summary.json"
	summary_payload = (unit_summary_json).read_text(encoding="utf-8")
	assert '"upsampling"' in summary_payload
	assert '"applied": true' in summary_payload
	assert '"effective_sampling_rate_hz": 100000.0' in summary_payload

	templates_summary_json = well_out_dir / "templates_outputs" / "templates_summary.json"
	templates_summary_payload = templates_summary_json.read_text(encoding="utf-8")
	assert '"upsampling_decisions_by_unit"' in templates_summary_payload
	assert '"execution_upsampling"' in templates_summary_payload


def test_run_templates_stage_passes_effective_sampling_rate_to_timing_renderers(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")

	def _fake_materialize(*, templates_out_dir: Path, **kwargs):
		_ = kwargs
		merged_dir = templates_out_dir / "templates" / "merged" / "unit_94"
		full_dir = templates_out_dir / "templates" / "full" / "unit_94"
		merged_dir.mkdir(parents=True, exist_ok=True)
		full_dir.mkdir(parents=True, exist_ok=True)
		t = np.vstack([np.sin(np.linspace(-1.0, 1.0, 40)), np.cos(np.linspace(-1.0, 1.0, 40))])
		locs = np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=float)
		np.save(merged_dir / "merged_contributing_template.npy", t)
		np.save(merged_dir / "merged_contributing_channel_locations.npy", locs)
		np.save(full_dir / "full_template.npy", t)
		np.save(full_dir / "full_channel_locations_xy.npy", locs)
		return (
			merged_dir.parent,
			full_dir.parent,
			{
				94: {
					"enabled": True,
					"applied": True,
					"factor": 10,
					"method": "sinc",
					"raw_hz": 10000.0,
					"analyzer_hz": 10000.0,
					"target_hz": 100000.0,
					"skip_reason": None,
				}
			},
		)

	received_hz: dict[str, float | None] = {
		"circles": None,
		"footprint_latency": None,
		"topographical_latency": None,
		"propagation": None,
	}

	def _fake_template_plot(**kwargs):
		_ = kwargs
		return {}

	def _fake_template_circles_plot(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received_hz["circles"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	def _fake_overlay(**kwargs):
		_ = kwargs
		return {}

	def _fake_amp_map(**kwargs):
		_ = kwargs
		return {}

	def _fake_latency_map(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received_hz["footprint_latency"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	def _fake_topo_amp(**kwargs):
		_ = kwargs
		return {}

	def _fake_topo_latency(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received_hz["topographical_latency"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	def _fake_propagation(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received_hz["propagation"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	monkeypatch.setattr(
		"axon_recon.pipeline.stages.templates.runner.materialize_templates_from_spikeinterface",
		_fake_materialize,
	)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", _fake_template_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", _fake_template_circles_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", _fake_overlay)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", _fake_amp_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", _fake_latency_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", _fake_topo_amp)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", _fake_topo_latency)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", _fake_propagation)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"

	assert received_hz["circles"] == 100_000.0
	assert received_hz["footprint_latency"] == 100_000.0
	assert received_hz["topographical_latency"] == 100_000.0
	assert received_hz["propagation"] == 100_000.0


def test_force_replot_reuses_persisted_sampling_metadata_for_timing_renderers(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	unit_summary_json = well_out_dir / "templates_outputs" / "units" / "0094" / "unit_templates_summary.json"
	unit_summary_json.parent.mkdir(parents=True, exist_ok=True)
	unit_summary_json.write_text(
		"""{
	  "unit_id": 94,
	  "status": "ok",
	  "upsampling": {
	    "enabled": true,
	    "applied": true,
	    "raw_hz": 10000.0,
	    "analyzer_hz": 10000.0,
	    "target_hz": 100000.0,
	    "skip_reason": null
	  },
	  "effective_sampling_rate_hz": 100000.0,
	  "outputs": {}
	}
	""",
		encoding="utf-8",
	)

	received: dict[str, float | None] = {"prop": None}

	def _fake_template_plot(**kwargs):
		_ = kwargs
		return {}

	def _fake_template_circles_plot(**kwargs):
		_ = kwargs
		return {}

	def _fake_overlay(**kwargs):
		_ = kwargs
		return {}

	def _fake_amp_map(**kwargs):
		_ = kwargs
		return {}

	def _fake_latency_map(**kwargs):
		_ = kwargs
		return {}

	def _fake_topo_amp(**kwargs):
		_ = kwargs
		return {}

	def _fake_topo_latency(**kwargs):
		_ = kwargs
		return {}

	def _fake_propagation(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received["prop"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", _fake_template_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", _fake_template_circles_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", _fake_overlay)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", _fake_amp_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", _fake_latency_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", _fake_topo_amp)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", _fake_topo_latency)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", _fake_propagation)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		force_replot=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"
	assert received["prop"] == 100_000.0


def test_force_replot_infers_sampling_rate_from_execution_when_metadata_missing(tmp_path: Path, monkeypatch) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	received: dict[str, float | None] = {"prop": None}

	def _fake_template_plot(**kwargs):
		_ = kwargs
		return {}

	def _fake_template_circles_plot(**kwargs):
		_ = kwargs
		return {}

	def _fake_overlay(**kwargs):
		_ = kwargs
		return {}

	def _fake_amp_map(**kwargs):
		_ = kwargs
		return {}

	def _fake_latency_map(**kwargs):
		_ = kwargs
		return {}

	def _fake_topo_amp(**kwargs):
		_ = kwargs
		return {}

	def _fake_topo_latency(**kwargs):
		_ = kwargs
		return {}

	def _fake_propagation(*, probe_geometry=None, **kwargs):
		_ = kwargs
		received["prop"] = None if probe_geometry is None else probe_geometry.sampling_rate_hz
		return {}

	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_plot", _fake_template_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_circles_plot", _fake_template_circles_plot)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_template_wf_overlay", _fake_overlay)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_amplitude_map", _fake_amp_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_footprint_latency_map", _fake_latency_map)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_amplitude_footprint", _fake_topo_amp)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_topographical_latency_footprint", _fake_topo_latency)
	monkeypatch.setattr("axon_recon.pipeline.stages.templates.runner.render_propagation_plot", _fake_propagation)

	inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		execution_upsampling=TimeUpsampleConfig(enabled=True, factor=10, method="sinc"),
		probe_geometry=ProbeGeometryConfig(sampling_rate_hz=10_000.0),
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=False, write_svg=False),
			template_circles=TemplateCirclesPlotConfig(write_png=False, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=False),
		),
		reports=ReportsConfig(
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=False),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		force_replot=True,
		n_jobs=1,
	)

	result = run_templates_stage(inputs)
	assert len(result.units) == 1
	assert result.units[0].status == "ok"
	assert received["prop"] == 100_000.0


def test_run_templates_stage_force_replot_rerenders_visual_outputs(tmp_path: Path) -> None:
	output_root = tmp_path / "outputs"
	h5_path = tmp_path / "dataset.h5"
	h5_path.write_text("", encoding="utf-8")

	well_out_dir = compute_mea_analysis_output_dir(output_root=output_root, data_file=h5_path, well="well000")
	_make_templates_artifacts(well_out_dir)

	first_inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=True),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(enabled=False),
			replot_from_disk=False,
			time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="linear"),
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=True),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=True,
		n_jobs=1,
	)
	first_result = run_templates_stage(first_inputs)
	assert len(first_result.units) == 1
	assert first_result.units[0].status == "ok"

	template_png = well_out_dir / "templates_outputs" / "units" / "0094" / "template.png"
	assert template_png.exists()
	mtime_before = template_png.stat().st_mtime_ns

	time.sleep(0.02)

	second_inputs = TemplatesInputs(
		h5_path=h5_path,
		stream_id="well000",
		mea_output_root=output_root,
		output_rel_root="templates_outputs",
		per_unit_outputs=PerUnitTemplatesOutputsConfig(
			unit_reldir="units/{unit_id:04d}/",
			template=TemplatePlotConfig(write_png=True, write_svg=False),
			template_wf_overlay=TemplateWaveformOverlayConfig(write_pdf=False, write_png=True),
		),
		reports=ReportsConfig(
			plot_multi_source_pdf=MultiSourcePdfReportConfig(enabled=False),
			replot_from_disk=True,
			time_upsample=TimeUpsampleConfig(enabled=False, factor=1, method="linear"),
			wf_overlay_grid=WfOverlayGridReportConfig(write_pdf=False, write_png=True),
		),
		unit_ids=[94],
		require_curated_units=False,
		force_restart=False,
		force_replot=True,
		n_jobs=1,
	)
	second_result = run_templates_stage(second_inputs)
	assert len(second_result.units) == 1
	assert second_result.units[0].status == "ok"

	mtime_after = template_png.stat().st_mtime_ns
	assert mtime_after > mtime_before
