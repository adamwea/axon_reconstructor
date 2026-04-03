from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.pipeline.stages.reconstruct.config import load_reconstruction_inputs_from_runtime


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
			global_heatmap_defaults:
			  default:
			    color_bar:
			      show_ticks: [1, 5, dynamic_high]
			      low_color: navy
			      mid_color: ivory
			      high_color: crimson
			      scale: log
			stages:
			  reconstruct:
			    inputs:
			      load_assets_from_v2pipeline_tempaltes_stage: true
			    execution:
			      force_restart: false
			    outputs:
			      output_rel_root: recon_outputs
			      cleanup_failed_unit_outputs: true
			      failed_units_summary_relpath: reports/failed_units.json
			      reports:
			        overwrite_on_unit_rerun: true
			        grids:
			          circle_recon_grid:
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
			              mode: direct_replot
			              dpi: 420
			              subplot_background_color: black
			              figure_background_color: black
			      write_summary: true
			      summary_relpath: reports/reconstruction_summary
			      summary_grid_ncols: 3
			      write_report_md: true
			      report_md_relpath: reports/reconstruction_report.md
			      amplitude_map:
			        write_png: true
			        relpath: maps/from_stage_block
			        panel_background_color: black
			        color_bar:
			          location: bottomleft
			          show_ticks: [2, 4, dynamic_high]
			          linear_cap_rounding_step: 5
			      per_unit_outputs:
			        write_gtr_pkl: true
			        write_gtr_json: true
			        amplitude_map_png_relpath: maps/amplitude_map.png
			        recon_plots:
			          circle_recon:
			            display:
			              base: template_circles
			              channel_scope: nodes_and_branches
			              zoom_padding_percent: 12
			              force_center_soma: true
			              branch_scope: raw
			              unique_color_per_branch: true
			              show_branch_labels: true
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
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_reconstruction_inputs_from_runtime(config_path=str(runtime_path), unit_id_override=94)
	assert inputs.stream_id == "well001"
	assert inputs.output_rel_root == "recon_outputs"
	assert inputs.write_summary_png is True
	assert inputs.summary_png_relpath == "reports/reconstruction_summary.png"
	assert inputs.summary_grid_ncols == 3
	assert inputs.write_report_md is True
	assert inputs.report_md_relpath == "reports/reconstruction_report.md"
	assert inputs.cleanup_failed_unit_outputs is True
	assert inputs.failed_units_summary_relpath == "reports/failed_units.json"
	assert inputs.per_unit_outputs.write_gtr_pkl is True
	assert inputs.per_unit_outputs.write_gtr_json is True
	assert inputs.per_unit_outputs.template_source == "square"
	assert inputs.per_unit_outputs.write_amplitude_map_png is True
	assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/amplitude_map.png"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.colorbar_location == "bottomleft"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.show_ticks == (2, 4, "dynamic_high")
	assert inputs.per_unit_outputs.amplitude_map_heatmap.low_color == "navy"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.scale == "log"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.linear_cap_rounding_step == 5.0
	circle = inputs.per_unit_outputs.circle_recon
	assert circle.display.base == "template_circles"
	assert circle.display.channel_scope == "nodes_and_branches"
	assert circle.display.zoom_padding_percent == 12.0
	assert circle.display.force_center_soma is True
	assert circle.display.branch_scope == "raw"
	assert circle.display.unique_color_per_branch is True
	assert circle.display.show_branch_labels is True
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
	assert inputs.reports.grids.circle_recon_grid.write_png is True
	assert inputs.reports.grids.circle_recon_grid.write_svg is True
	assert inputs.reports.overwrite_on_unit_rerun is True
	assert inputs.reports.grids.circle_recon_grid.png_relpath == "reports/circle_recon_grid.png"
	assert inputs.reports.grids.circle_recon_grid.svg_relpath == "reports/circle_recon_grid.svg"
	assert inputs.reports.grids.circle_recon_grid.keep_temp_svg is True
	assert inputs.reports.grids.circle_recon_grid.show_title is False
	assert inputs.reports.grids.circle_recon_grid.render_mode == "direct_replot"
	assert inputs.reports.grids.circle_recon_grid.dpi == 420.0
	assert inputs.load_assets_from_v2pipeline_templates_stage is True
	assert inputs.unit_ids == [94]


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
	assert inputs.per_unit_outputs.write_amplitude_map_png is True
	assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/legacy_stage_amp.png"
	heat = inputs.per_unit_outputs.amplitude_map_heatmap
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
	assert inputs.per_unit_outputs.template_source == "full_from_merged"


def test_load_config_reads_canonical_axon_velocity_block_with_legacy_fallback(tmp_path: Path) -> None:
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
			    av:
			      detect_threshold: 0.4
			      min_path_points: 5
			    axon_velocity:
			      detect_threshold: 0.0001
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
