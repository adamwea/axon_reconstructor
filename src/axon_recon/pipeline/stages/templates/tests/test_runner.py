from __future__ import annotations

from pathlib import Path
import shutil
import time

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.templates.models.inputs import (
	FootprintGridsReportConfig,
	FootprintMapGridReportConfig,
	FootprintMapConfig,
	FootprintPlotsConfig,
	MultiSourcePdfReportConfig,
	PerUnitTemplatesOutputsConfig,
	PropagationPlotConfig,
	ReportsConfig,
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
	lat_grid_png = well_out_dir / "templates_outputs" / "reports" / "latency_map_grid.png"
	assert amp_grid_png.exists()
	assert lat_grid_png.exists()
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
