from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.pipeline.stages.reconstruct.config import load_reconstruction_inputs_from_runtime
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def test_load_config_reads_runtime_and_data(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			    wells:
			      - well_id: well001
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    output_rel_root: recon_outputs
			    cleanup_failed_unit_outputs: true
			    failed_units_summary_relpath: reports/failed_units.json
			    report_sort_by: template density
			    overwrite_report_outputs_on_unit_rerun: true
			    resources:
			      max_plotting_concurrency: 2
			    debug_prints: true
			    phase_sequence: [generate_gtrs, plot_recons, report_recons, report_recon_grid, report_summaries]
			    debug_mode:
			      enabled: true
			      limit_datasets: 2
			      limit_wells: 4
			      limit_wells_per_dataset: 1
			    execution:
			      force_restart: false
			    phases:
			      generate_gtrs:
			        outputs:
			          template_source: merged
			          write_gtr_pkl: true
			          write_detection_filter_json: true
			          detection_filter_relpath: filters/detect.json
			          write_gtr_json: true
			          diagnostic_figs:
			            channel_selection:
			              write_png: true
			              invert_y_axis: false
			              relpath: figures/channel_selection
			            axon_reconstruction:
			              write_png: true
			              write_svg: true
			              invert_y_axis: false
			              relpath: figures/axon_reconstruction
			      plot_recons:
			        outputs:
			          amplitude_map:
			            write_png: true
			            relpath: maps/from_stage_block
			            invert_y_axis: false
			            panel_background_color: black
			            color_bar:
			              low_color: navy
			              mid_color: ivory
			              high_color: crimson
			              scale: log
			              location: bottomleft
			              show_ticks: [2, 4, dynamic_high]
			              linear_cap_rounding_step: 5
			          circle_recon:
			            display:
			              base: template_circles
			              channel_scope: nodes_and_branches
			              zoom_padding_percent: 12
			              invert_y_axis: false
			              force_center_soma: true
			              branch_scope: raw
			              unique_color_per_branch: true
			              show_branch_labels: true
			              show_branch_legend: true
			              color_scheme: tab20
			              node_outline_color: white
			              node_outline_linewidth: 2.5
			              branch_outline_color: white
			              branch_outline_linewidth: 1.5
			              node_inline_linewidth_pt: 0.42
			              branch_linewidth_pt: 1.1
			            output:
			              write_png: true
			              write_svg: true
			              relpath: maps/circle_recon
			              dpi: 420
			      report_recons:
			        summary_png:
			          write: true
			          relpath: reports/reconstruction_summary
			          grid_ncols: 3
			        report_md:
			          write: true
			          relpath: reports/reconstruction_report.md
			      report_recon_grid:
			        output:
			          write_pdf: false
			          pdf_relpath: reports/circle_recon_grid.pdf
			          write_png: true
			          png_relpath: reports/circle_recon_grid.png
			          write_svg: true
			          svg_relpath: reports/circle_recon_grid.svg
			          keep_temp_svg: true
			          temp_svg_relpath: reports/circle_recon_grid__temp.svg
			        display:
			          show_title: false
			        render:
			          dpi: 420
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path), unit_id_override=94)
	assert inputs.stream_id == "well001"
	assert inputs.debug_prints is True
	assert inputs.max_plotting_concurrency == 2
	assert inputs.phase_sequence == ("generate_gtrs", "plot_recons", "report_recons", "report_recon_grid", "report_summaries")
	assert inputs.output_rel_root == "recon_outputs"
	assert inputs.report_sort_by == "template_density"
	assert inputs.overwrite_report_outputs_on_unit_rerun is True
	assert inputs.phases.report_recons.summary_png.write is True
	assert inputs.phases.report_recons.summary_png.relpath == "reports/reconstruction_summary.png"
	assert inputs.phases.report_recons.summary_png.grid_ncols == 3
	assert inputs.phases.report_recons.report_md.write is True
	assert inputs.phases.report_recons.report_md.relpath == "reports/reconstruction_report.md"
	assert inputs.cleanup_failed_unit_outputs is True
	assert inputs.failed_units_summary_relpath == "reports/failed_units.json"
	assert inputs.phases.generate_gtrs.outputs.template_source == "merged"
	assert inputs.phases.generate_gtrs.outputs.write_gtr_pkl is True
	assert inputs.phases.generate_gtrs.outputs.write_detection_filter_json is True
	assert inputs.phases.generate_gtrs.outputs.detection_filter_relpath == "filters/detect.json"
	assert inputs.phases.generate_gtrs.outputs.write_gtr_json is True
	assert inputs.phases.generate_gtrs.outputs.channel_selection_figure.write_png is True
	assert inputs.phases.generate_gtrs.outputs.channel_selection_figure.invert_y_axis is False
	assert inputs.phases.generate_gtrs.outputs.channel_selection_figure.relpath == "figures/channel_selection"
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.write_png is True
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.write_svg is True
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.invert_y_axis is False
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.relpath == "figures/axon_reconstruction"
	assert inputs.per_unit_outputs.write_gtr_pkl is True
	assert inputs.per_unit_outputs.write_detection_filter_json is True
	assert inputs.per_unit_outputs.detection_filter_relpath == "filters/detect.json"
	assert inputs.per_unit_outputs.write_gtr_json is True
	assert inputs.per_unit_outputs.template_source == "merged"
	assert inputs.per_unit_outputs.channel_selection_figure.write_png is True
	assert inputs.per_unit_outputs.channel_selection_figure.invert_y_axis is False
	assert inputs.per_unit_outputs.channel_selection_figure.relpath == "figures/channel_selection"
	assert inputs.per_unit_outputs.axon_reconstruction_figure.write_png is True
	assert inputs.per_unit_outputs.axon_reconstruction_figure.write_svg is True
	assert inputs.per_unit_outputs.axon_reconstruction_figure.invert_y_axis is False
	assert inputs.per_unit_outputs.axon_reconstruction_figure.relpath == "figures/axon_reconstruction"
	assert inputs.phases.plot_recons.outputs.amplitude_map.write_png is True
	assert inputs.phases.plot_recons.outputs.amplitude_map.png_relpath == "maps/from_stage_block.png"
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.colorbar_location == "bottomleft"
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.invert_y_axis is False
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.show_ticks == (2, 4, "dynamic_high")
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.low_color == "navy"
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.scale == "log"
	assert inputs.phases.plot_recons.outputs.amplitude_map.heatmap.linear_cap_rounding_step == 5.0
	assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/from_stage_block.png"
	circle = inputs.phases.plot_recons.outputs.circle_recon
	assert circle.display.base == "template_circles"
	assert circle.display.channel_scope == "nodes_and_branches"
	assert circle.display.zoom_padding_percent == 12.0
	assert circle.display.invert_y_axis is False
	assert circle.display.force_center_soma is True
	assert circle.display.branch_scope == "raw"
	assert circle.display.unique_color_per_branch is True
	assert circle.display.show_branch_labels is True
	assert circle.display.show_branch_legend is True
	assert circle.display.color_scheme == "tab20"
	assert circle.display.node_outline_color == "white"
	assert circle.display.node_outline_linewidth == 2.5
	assert circle.display.branch_outline_color == "white"
	assert circle.display.branch_outline_linewidth == 1.5
	assert circle.display.node_border_linewidth == 0.42
	assert circle.display.edge_linewidth == 1.1
	assert circle.output.write_png is True
	assert circle.output.write_svg is True
	assert circle.output.relpath == "maps/circle_recon"
	assert circle.output.dpi == 420.0
	assert inputs.phases.report_recon_grid.output.write_png is True
	assert inputs.phases.report_recon_grid.output.write_svg is True
	assert inputs.phases.report_recon_grid.output.png_relpath == "reports/circle_recon_grid.png"
	assert inputs.phases.report_recon_grid.output.svg_relpath == "reports/circle_recon_grid.svg"
	assert inputs.phases.report_recon_grid.output.keep_temp_svg is True
	assert inputs.phases.report_recon_grid.display.show_title is False
	assert inputs.phases.report_recon_grid.render.dpi == 420.0
	assert inputs.unit_ids == [94]


def test_load_config_reconstruct_populates_templates_inputs_from_debug_runtime() -> None:
	repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "debug" / "debug.runtime.yml").exists())
	inputs = load_reconstruction_inputs_from_runtime(config_path=str(repo_root / "debug" / "debug.runtime.yml"))
	assert inputs.templates_inputs is not None
	assert isinstance(inputs.templates_inputs, TemplatesInputs)
	assert inputs.debug_prints is False
	assert inputs.phase_sequence == (
		"templates_analyzers",
		"templates_build_templates",
		"templates_plot_templates",
		"templates_report_templates",
		"generate_gtrs",
		"plot_recons",
		"plot_branch_propagations",
		"plot_branch_velocities",
		"plot_unit_summary",
		"report_recons",
		"report_recon_grid",
		"report_full_chip_layout",
		"report_summaries",
		"clear_templates_cache",
	)


def test_load_config_reconstruct_reads_runtime_unit_ids(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    execution:
			      unit_ids: [3, 7, 9, 7]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.unit_ids == [3, 7, 9]


def test_load_config_reconstruct_debug_limits_flow_to_templates_inputs(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  templates:
			    debug_mode:
			      unit_limit: 9
			      limit_segments: 8
			  reconstruct:
			    unit_limit: 99
			    debug_mode:
			      unit_limit: 2
			      limit_segments: 3
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.unit_limit == 2
	assert inputs.limit_segments == 3
	assert inputs.templates_inputs is not None
	assert inputs.templates_inputs.unit_limit == 2
	assert inputs.templates_inputs.limit_segments == 3

	inputs_override = load_reconstruction_inputs_from_runtime(
		config_path=str(runtime_path),
		unit_limit_override=4,
		limit_segments_override=5,
	)
	assert inputs_override.unit_limit == 4
	assert inputs_override.limit_segments == 5
	assert inputs_override.templates_inputs is not None
	assert inputs_override.templates_inputs.unit_limit == 4
	assert inputs_override.templates_inputs.limit_segments == 5


def test_load_config_reconstruct_prefers_unit_ids_override(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    execution:
			      unit_ids: [3, 7, 9]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(
		config_path=str(runtime_path),
		unit_ids_override=[44, 50, 44],
	)
	assert inputs.unit_ids == [44, 50]


def test_load_config_reconstruct_parses_branch_plot_phase_and_shared_branch_colors(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    branch_colors:
			      unique_color_per_branch: false
			      color_scheme: Set1
			    phases:
			      plot_recons:
			        enable: true
			        outputs:
			          circle_recon:
			            display:
			              branch_scope: raw
			      plot_branch_propagations:
			        enable: true
			        branch_scope: raw
			        display:
			          figsize: [7, 5]
			          total_width: 18
			          sort_templates: true
			          show_title: false
			          invert_y_axis: false
			        output:
			          write_png: true
			          relpath: branch_qc/propagations
			          manifest_relpath: reports/branch_propagations_manifest.json
			      plot_branch_velocities:
			        enable: true
			        branch_scope: clean
			        display:
			          figsize: [8, 4]
			          show_title: false
			          title_fontsize: 14
			          axis_label_fontsize: 12
			          tick_label_fontsize: 10.5
			          units_only_axis_labels: false
			          show_legend: false
			          legend_fontsize: 11
			        output:
			          write_png: true
			          relpath: branch_qc/velocities
			          manifest_relpath: reports/branch_velocities_manifest.json
			      plot_unit_summary:
			        enable: true
			        display:
			          show_title: true
			          show_summary_unit_label: true
			          summary_unit_label_fontsize: 28
			          summary_unit_label_x_frac: 0.03
			          summary_unit_label_y_frac: 0.97
			          recon_show_unit_label: false
			          recon_show_branch_legend: false
			          velocity_show_title: false
			          show_velocity_legend: true
			          reserve_velocity_legend_space: true
			          velocity_legend_width: 2.75
			          top_row_panel_gap_width: 0.85
			          top_row_height: 7.25
			          propagation_row_height: 5.5
			          circle_panel_width: 6.25
			          velocity_panel_width: 5.25
			          propagation_panel_width: 3.0
			          recon_x_offset_frac: 0.01
			          recon_y_offset_frac: -0.02
			          velocity_x_offset_frac: -0.03
			          velocity_y_offset_frac: 0.04
			          propagation_x_offset_frac: 0.02
			          propagation_y_offset_frac: -0.01
			        output:
			          write_png: true
			          write_svg: true
			          relpath: reports/unit_summary
			          dpi: 240
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.branch_colors.unique_color_per_branch is False
	assert inputs.branch_colors.color_scheme == "Set1"
	assert inputs.per_unit_outputs.circle_recon.display.unique_color_per_branch is False
	assert inputs.per_unit_outputs.circle_recon.display.color_scheme == "Set1"
	assert inputs.phases.plot_branch_propagations.enabled is True
	assert inputs.phases.plot_branch_propagations.branch_scope == "raw"
	assert inputs.phases.plot_branch_propagations.display.figsize == (7.0, 5.0)
	assert inputs.phases.plot_branch_propagations.display.total_width == 18.0
	assert inputs.phases.plot_branch_propagations.display.sort_templates is True
	assert inputs.phases.plot_branch_propagations.display.show_title is False
	assert inputs.phases.plot_branch_propagations.display.invert_y_axis is False
	assert inputs.phases.plot_branch_propagations.output.write_png is True
	assert inputs.phases.plot_branch_propagations.output.relpath == "branch_qc/propagations"
	assert inputs.phases.plot_branch_propagations.output.manifest_relpath == "reports/branch_propagations_manifest.json"
	assert inputs.phases.plot_branch_velocities.enabled is True
	assert inputs.phases.plot_branch_velocities.branch_scope == "clean"
	assert inputs.phases.plot_branch_velocities.display.figsize == (8.0, 4.0)
	assert inputs.phases.plot_branch_velocities.display.show_title is False
	assert inputs.phases.plot_branch_velocities.display.title_fontsize == 14.0
	assert inputs.phases.plot_branch_velocities.display.axis_label_fontsize == 12.0
	assert inputs.phases.plot_branch_velocities.display.tick_label_fontsize == 10.5
	assert inputs.phases.plot_branch_velocities.display.units_only_axis_labels is False
	assert inputs.phases.plot_branch_velocities.display.show_legend is False
	assert inputs.phases.plot_branch_velocities.display.legend_fontsize == 11.0
	assert inputs.phases.plot_branch_velocities.output.relpath == "branch_qc/velocities"
	assert inputs.phases.plot_branch_velocities.output.manifest_relpath == "reports/branch_velocities_manifest.json"
	assert inputs.phases.plot_unit_summary.enabled is True
	assert inputs.phases.plot_unit_summary.display.show_title is True
	assert inputs.phases.plot_unit_summary.display.show_summary_unit_label is True
	assert inputs.phases.plot_unit_summary.display.summary_unit_label_fontsize == 28.0
	assert inputs.phases.plot_unit_summary.display.summary_unit_label_x_frac == 0.03
	assert inputs.phases.plot_unit_summary.display.summary_unit_label_y_frac == 0.97
	assert inputs.phases.plot_unit_summary.display.recon_show_unit_label is False
	assert inputs.phases.plot_unit_summary.display.recon_show_branch_legend is False
	assert inputs.phases.plot_unit_summary.display.velocity_show_title is False
	assert inputs.phases.plot_unit_summary.display.show_velocity_legend is True
	assert inputs.phases.plot_unit_summary.display.reserve_velocity_legend_space is True
	assert inputs.phases.plot_unit_summary.display.velocity_legend_width == 2.75
	assert inputs.phases.plot_unit_summary.display.top_row_panel_gap_width == 0.85
	assert inputs.phases.plot_unit_summary.display.top_row_height == 7.25
	assert inputs.phases.plot_unit_summary.display.propagation_row_height == 5.5
	assert inputs.phases.plot_unit_summary.display.circle_panel_width == 6.25
	assert inputs.phases.plot_unit_summary.display.velocity_panel_width == 5.25
	assert inputs.phases.plot_unit_summary.display.propagation_panel_width == 3.0
	assert inputs.phases.plot_unit_summary.display.recon_x_offset_frac == 0.01
	assert inputs.phases.plot_unit_summary.display.recon_y_offset_frac == -0.02
	assert inputs.phases.plot_unit_summary.display.velocity_x_offset_frac == -0.03
	assert inputs.phases.plot_unit_summary.display.velocity_y_offset_frac == 0.04
	assert inputs.phases.plot_unit_summary.display.propagation_x_offset_frac == 0.02
	assert inputs.phases.plot_unit_summary.display.propagation_y_offset_frac == -0.01
	assert inputs.phases.plot_unit_summary.output.write_png is True
	assert inputs.phases.plot_unit_summary.output.write_svg is True
	assert inputs.phases.plot_unit_summary.output.relpath == "reports/unit_summary"
	assert inputs.phases.plot_unit_summary.output.dpi == 240.0


def test_load_config_reconstruct_parses_full_chip_layout_phase(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    phases:
			      report_full_chip_layout:
			        enable: true
			        branch_scope: clean
			        unit_colors:
			          strategy: colormap
			          color_scheme: tab20
			        display:
			          figsize: [12, 7]
			          show_title: false
			          invert_y_axis: false
			          alpha: 0.55
			          linewidth: 2.0
			          show_legend: true
			          legend_fontsize: 7
			          legend_ncols: 2
			          draw_chip_outline: false
			          background_color: black
			        output:
			          write_png: true
			          write_svg: true
			          relpath: reports/chip_layout
			          manifest_relpath: reports/chip_layout_manifest.json
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	phase = inputs.phases.report_full_chip_layout
	assert phase.enabled is True
	assert phase.branch_scope == "clean"
	assert phase.unit_colors.strategy == "colormap"
	assert phase.unit_colors.color_scheme == "tab20"
	assert phase.display.figsize == (12.0, 7.0)
	assert phase.display.show_title is False
	assert phase.display.invert_y_axis is False
	assert phase.display.alpha == 0.55
	assert phase.display.linewidth == 2.0
	assert phase.display.show_legend is True
	assert phase.display.legend_fontsize == 7.0
	assert phase.display.legend_ncols == 2
	assert phase.display.draw_chip_outline is False
	assert phase.display.background_color == "black"
	assert phase.output.write_png is True
	assert phase.output.write_svg is True
	assert phase.output.relpath == "reports/chip_layout"
	assert phase.output.manifest_relpath == "reports/chip_layout_manifest.json"


def test_load_config_reconstruct_legacy_stage_block_without_global_defaults(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    outputs:
			      amplitude_map:
			        write_png: true
			        relpath: maps/legacy_stage_amp
			        panel_background_color: white
			        color_bar:
			          low_color: teal
			          show_ticks: [1, 3, dynamic_high]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.phases.plot_recons.outputs.amplitude_map.write_png is True
	assert inputs.phases.plot_recons.outputs.amplitude_map.png_relpath == "maps/legacy_stage_amp.png"
	assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/legacy_stage_amp.png"
	heat = inputs.phases.plot_recons.outputs.amplitude_map.heatmap
	assert heat.background == "white"
	assert heat.low_color == "teal"
	assert heat.show_ticks == (1, 3, "dynamic_high")


def test_load_config_accepts_full_from_merged_template_source(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    outputs:
			      per_unit_outputs:
			        template_source: full_from_merged
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.phases.generate_gtrs.outputs.template_source == "full_from_merged"
	assert inputs.per_unit_outputs.template_source == "full_from_merged"


def test_load_config_reads_canonical_axon_velocity_block(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    axon_velocity:
			      detect_threshold: 0.0001
			      min_path_points: 5
			      n_neighbors: 8
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert float(inputs.axon_velocity_params["detect_threshold"]) == 0.0001
	assert int(inputs.axon_velocity_params["min_path_points"]) == 5
	assert int(inputs.axon_velocity_params["n_neighbors"]) == 8


def test_load_config_reads_reconstruct_phase_blocks_and_overrides_stage_defaults(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "runtime.yml"
	runtime_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  reconstruct:
			    axon_velocity:
			      detect_threshold: 0.4
			      min_path_points: 5
			    outputs:
			      per_unit_outputs:
			        write_branches_json: false
			        template_source: square
			    phases:
			      generate_gtrs:
			        enable: false
			        summary_json_relpath: context/gtrs_phase.json
			        resources:
			          max_plotting_concurrency: 3
			          unit_procs: 3
			          unit_batch_size: 11
			        outputs:
			          template_source: merged
			          write_branches_json: true
			          write_all_filters_json: true
			          all_filters_relpath: phase/all_filters.json
			          write_gtr_json: true
			          gtr_json_relpath: phase/gtr.json
			          diagnostic_figs:
			            channel_selection:
			              write_png: true
			              relpath: phase/channel_selection
			            axon_reconstruction:
			              write_svg: true
			              relpath: phase/axon_reconstruction
			        axon_velocity:
			          enabled: true
			          params:
			            detect_threshold: 0.0001
			            n_neighbors: 8
			      plot_recons:
			        enabled: true
			        summary_json_relpath: context/plot_phase.json
			        outputs:
			          circle_recon:
			            display:
			              channel_scope: selected_channels
			              color_scheme: Set1
			            output:
			              write_png: true
			              write_svg: false
			              relpath: phase_circle
			      report_recons:
			        enable: true
			        summary_json_relpath: context/report_phase.json
			        av_recons:
			          write_pdf: true
			          pdf_relpath: reports/av_recons.pdf
			      report_recon_grid:
			        enabled: true
			        summary_json_relpath: context/report_recon_grid_phase.json
			        output:
			          write_png: true
			          png_relpath: reports/circle_recon_grid.png
			        display:
			          show_title: false
			        render:
			          dpi: 420
			      report_summaries:
			        enable: true
			        summary_json_relpath: context/report_summaries_phase.json
			        write_pdf: true
			        pdf_relpath: reports/reconstruct_summary_deck.pdf
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.phases.generate_gtrs.enabled is False
	assert inputs.max_plotting_concurrency == 3
	assert inputs.phases.generate_gtrs.summary_json_relpath == "context/gtrs_phase.json"
	assert inputs.phases.generate_gtrs.unit_procs == 3
	assert inputs.phases.generate_gtrs.unit_batch_size == 11
	assert inputs.phases.generate_gtrs.outputs.template_source == "merged"
	assert inputs.phases.generate_gtrs.outputs.write_branches_json is True
	assert inputs.phases.generate_gtrs.outputs.write_all_filters_json is True
	assert inputs.phases.generate_gtrs.outputs.all_filters_relpath == "phase/all_filters.json"
	assert inputs.phases.generate_gtrs.outputs.write_gtr_json is True
	assert inputs.phases.generate_gtrs.outputs.gtr_json_relpath == "phase/gtr.json"
	assert inputs.phases.generate_gtrs.outputs.channel_selection_figure.write_png is True
	assert inputs.phases.generate_gtrs.outputs.channel_selection_figure.relpath == "phase/channel_selection"
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.write_svg is True
	assert inputs.phases.generate_gtrs.outputs.axon_reconstruction_figure.relpath == "phase/axon_reconstruction"
	assert inputs.phases.plot_recons.enabled is True
	assert inputs.phases.plot_recons.summary_json_relpath == "context/plot_phase.json"
	assert inputs.phases.report_recons.enabled is True
	assert inputs.phases.report_recons.summary_json_relpath == "context/report_phase.json"
	assert inputs.phases.report_recons.av_recons.write_pdf is True
	assert inputs.phases.report_recons.av_recons.pdf_relpath == "reports/av_recons.pdf"
	assert inputs.phases.report_recon_grid.enabled is True
	assert inputs.phases.report_recon_grid.summary_json_relpath == "context/report_recon_grid_phase.json"
	assert inputs.phases.report_recon_grid.output.write_png is True
	assert inputs.phases.report_recon_grid.output.png_relpath == "reports/circle_recon_grid.png"
	assert inputs.phases.report_recon_grid.display.show_title is False
	assert inputs.phases.report_recon_grid.render.dpi == 420.0
	assert inputs.phases.report_summaries.enabled is True
	assert inputs.phases.report_summaries.summary_json_relpath == "context/report_summaries_phase.json"
	assert inputs.phases.report_summaries.write_pdf is True
	assert inputs.phases.report_summaries.pdf_relpath == "reports/reconstruct_summary_deck.pdf"
	assert float(inputs.axon_velocity_params["detect_threshold"]) == 0.0001
	assert int(inputs.axon_velocity_params["min_path_points"]) == 5
	assert int(inputs.axon_velocity_params["n_neighbors"]) == 8
	assert inputs.per_unit_outputs.template_source == "merged"
	assert inputs.per_unit_outputs.write_branches_json is True
	assert inputs.per_unit_outputs.write_all_filters_json is True
	assert inputs.per_unit_outputs.all_filters_relpath == "phase/all_filters.json"
	assert inputs.per_unit_outputs.write_gtr_json is True
	assert inputs.per_unit_outputs.gtr_json_relpath == "phase/gtr.json"
	assert inputs.per_unit_outputs.channel_selection_figure.write_png is True
	assert inputs.per_unit_outputs.channel_selection_figure.relpath == "phase/channel_selection"
	assert inputs.per_unit_outputs.axon_reconstruction_figure.write_svg is True
	assert inputs.per_unit_outputs.axon_reconstruction_figure.relpath == "phase/axon_reconstruction"
	assert inputs.per_unit_outputs.circle_recon.display.channel_scope == "selected_channels"
	assert inputs.per_unit_outputs.circle_recon.output.write_png is True
	assert inputs.per_unit_outputs.circle_recon.output.relpath == "phase_circle"
	assert inputs.per_unit_outputs.circle_recon.display.color_scheme == "Set1"
