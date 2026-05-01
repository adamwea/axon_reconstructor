from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models.inputs import ReportsConfig, TemplatesInputs
from ..models.results import TemplatesResult


def propagation_outputs_requested(propagation_plots: Any) -> bool:
	return any(
		bool(getattr(propagation_plots, field_name, False))
		for field_name in (
			"write_pdf",
			"write_png",
			"write_svg",
			"write_circles_template_numbered_png",
			"write_circles_template_numbered_svg",
			"write_propagation_2panel_png",
			"write_propagation_2panel_svg",
		)
	)


def excluded_plot_output_keys() -> tuple[str, ...]:
	return (
		"template_png",
		"template_svg",
		"template_wf_overlay_pdf",
		"template_wf_overlay_png",
		"extremum_ch_wf_overlay_pdf",
		"extremum_ch_wf_overlay_png",
		"footprint_amplitude_map_png",
		"footprint_amplitude_map_svg",
		"footprint_latency_map_png",
		"footprint_latency_map_svg",
		"topographical_amplitude_footprint_png",
		"topographical_amplitude_footprint_svg",
		"topographical_latency_footprint_png",
		"topographical_latency_footprint_svg",
		"quality_checks_multiple_negative_peaks_plot_png",
		"quality_checks_multiple_negative_peaks_plot_svg",
		"propagation_plot_pdf",
		"propagation_plot_png",
		"propagation_plot_svg",
		"circles_template_numbered_png",
		"circles_template_numbered_svg",
		"propagation_2panel_png",
		"propagation_2panel_svg",
		"propagation_plot_left_temp_svg",
		"propagation_plot_right_temp_svg",
		"propagation_plot_right_temp_png",
	)


def requested_plot_output_keys(per_unit_outputs: Any) -> tuple[str, ...]:
	requested: list[str] = []
	if bool(per_unit_outputs.template.write_png):
		requested.append("template_png")
	if bool(per_unit_outputs.template.write_svg):
		requested.append("template_svg")
	if bool(per_unit_outputs.template_circles.write_png):
		requested.append("template_circles_png")
	if bool(per_unit_outputs.template_circles.write_svg):
		requested.append("template_circles_svg")
	if bool(per_unit_outputs.template_wf_overlay.write_pdf):
		requested.append("template_wf_overlay_pdf")
	if bool(per_unit_outputs.template_wf_overlay.write_png):
		requested.append("template_wf_overlay_png")
	if bool(per_unit_outputs.footprint_plots.amplitude_map.write_png):
		requested.append("footprint_amplitude_map_png")
	if bool(per_unit_outputs.footprint_plots.amplitude_map.write_svg):
		requested.append("footprint_amplitude_map_svg")
	if bool(per_unit_outputs.footprint_plots.latency_map.write_png):
		requested.append("footprint_latency_map_png")
	if bool(per_unit_outputs.footprint_plots.latency_map.write_svg):
		requested.append("footprint_latency_map_svg")
	if bool(per_unit_outputs.topographical_footprints.amplitude.write_png):
		requested.append("topographical_amplitude_footprint_png")
	if bool(per_unit_outputs.topographical_footprints.amplitude.write_svg):
		requested.append("topographical_amplitude_footprint_svg")
	if bool(per_unit_outputs.topographical_footprints.latency.write_png):
		requested.append("topographical_latency_footprint_png")
	if bool(per_unit_outputs.topographical_footprints.latency.write_svg):
		requested.append("topographical_latency_footprint_svg")
	if bool(per_unit_outputs.propagation_plots.write_pdf):
		requested.append("propagation_plot_pdf")
	if bool(per_unit_outputs.propagation_plots.write_png):
		requested.append("propagation_plot_png")
	if bool(per_unit_outputs.propagation_plots.write_svg):
		requested.append("propagation_plot_svg")
	if bool(per_unit_outputs.propagation_plots.write_circles_template_numbered_png):
		requested.append("circles_template_numbered_png")
	if bool(per_unit_outputs.propagation_plots.write_circles_template_numbered_svg):
		requested.append("circles_template_numbered_svg")
	if bool(per_unit_outputs.propagation_plots.write_propagation_2panel_png):
		requested.append("propagation_2panel_png")
	if bool(per_unit_outputs.propagation_plots.write_propagation_2panel_svg):
		requested.append("propagation_2panel_svg")
	qc_plot = per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.plot
	if bool(qc_plot.write_png):
		requested.append("quality_checks_multiple_negative_peaks_plot_png")
	if bool(qc_plot.write_svg):
		requested.append("quality_checks_multiple_negative_peaks_plot_svg")
	return tuple(requested)


def _disable_reports_config(reports: ReportsConfig) -> ReportsConfig:
	return replace(
		reports,
		locations=replace(reports.locations, write_json=False, write_png=False, write_svg=False),
		wf_overlay_grid=replace(reports.wf_overlay_grid, write_pdf=False, write_png=False, write_svg=False),
		footprint_grids=replace(
			reports.footprint_grids,
			circles_map_grid=replace(
				reports.footprint_grids.circles_map_grid,
				write_pdf=False,
				write_png=False,
				write_svg=False,
			),
			amplitude_map_grid=replace(
				reports.footprint_grids.amplitude_map_grid,
				write_pdf=False,
				write_png=False,
				write_svg=False,
			),
			latency_map_grid=replace(
				reports.footprint_grids.latency_map_grid,
				write_pdf=False,
				write_png=False,
				write_svg=False,
			),
		),
		plot_multi_source_pdf=replace(reports.plot_multi_source_pdf, enabled=False),
	)


def build_plot_templates_phase_inputs(inputs: TemplatesInputs) -> TemplatesInputs:
	per_unit_outputs = replace(
		inputs.per_unit_outputs,
		template=replace(
			inputs.per_unit_outputs.template,
			write_png=False,
			write_svg=False,
		),
		quality_checks=replace(
			inputs.per_unit_outputs.quality_checks,
			check_for_multiple_peaks_at_channel_templates=replace(
				inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates,
				plot=replace(
					inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates.plot,
					write_png=False,
					write_svg=False,
				),
			),
		),
		template_circles=replace(
			inputs.per_unit_outputs.template_circles,
			show_propagation_order_labels=False,
		),
		template_wf_overlay=replace(
			inputs.per_unit_outputs.template_wf_overlay,
			write_pdf=False,
			write_png=False,
		),
		footprint_plots=replace(
			inputs.per_unit_outputs.footprint_plots,
			amplitude_map=replace(
				inputs.per_unit_outputs.footprint_plots.amplitude_map,
				write_png=False,
				write_svg=False,
			),
			latency_map=replace(
				inputs.per_unit_outputs.footprint_plots.latency_map,
				write_png=False,
				write_svg=False,
			),
		),
		topographical_footprints=replace(
			inputs.per_unit_outputs.topographical_footprints,
			amplitude=replace(
				inputs.per_unit_outputs.topographical_footprints.amplitude,
				write_png=False,
				write_svg=False,
			),
			latency=replace(
				inputs.per_unit_outputs.topographical_footprints.latency,
				write_png=False,
				write_svg=False,
			),
		),
		propagation_plots=replace(
			inputs.per_unit_outputs.propagation_plots,
			write_pdf=False,
			write_png=False,
			write_svg=False,
			write_circles_template_numbered_png=False,
			write_circles_template_numbered_svg=False,
			write_propagation_2panel_png=False,
			write_propagation_2panel_svg=False,
		),
	)
	phases = replace(
		inputs.phases,
		plot_templates=replace(inputs.phases.plot_templates, outputs=per_unit_outputs),
		per_unit_processing=replace(
			inputs.phases.per_unit_processing,
			plots=replace(inputs.phases.per_unit_processing.plots, outputs=per_unit_outputs),
		),
	)
	return replace(
		inputs,
		per_unit_outputs=per_unit_outputs,
		quality_checks_outputs=replace(
			inputs.quality_checks_outputs,
			check_for_multiple_peaks_at_channel_templates=replace(
				inputs.quality_checks_outputs.check_for_multiple_peaks_at_channel_templates,
				write_json=False,
			),
		),
		phases=phases,
		reports=_disable_reports_config(inputs.reports),
		force_restart=False,
		force_replot=True,
		force_replot_per_unit=False,
		force_rereport=False,
	)


def build_plot_templates_phase_summary(
	*,
	inputs: TemplatesInputs,
	result: TemplatesResult,
	skipped_units: list[Any] | None = None,
	duration_seconds: float,
) -> dict[str, Any]:
	skipped = list(skipped_units or [])
	ok_units = [unit.unit_id for unit in result.units if str(unit.status) == "ok"]
	failed_units = [
		{
			"unit_id": unit.unit_id,
			"error": unit.error,
		}
		for unit in result.units
		if str(unit.status) != "ok"
	]
	return {
		"phase": "plot_templates",
		"well_out_dir": str(result.well_out_dir),
		"templates_out_dir": str(result.templates_out_dir),
		"templates_summary_json": str(result.summary_json),
		"duration_seconds": float(duration_seconds),
		"unit_count": int(len(ok_units) + len(failed_units) + len(skipped)),
		"rendered_units": ok_units,
		"skipped_units": skipped,
		"failed_units": failed_units,
		"excluded_outputs": list(excluded_plot_output_keys()),
		"propagation_outputs_enabled": propagation_outputs_requested(inputs.per_unit_outputs.propagation_plots),
		"summary_json_relpath": str(inputs.phases.plot_templates.summary_json_relpath),
	}
