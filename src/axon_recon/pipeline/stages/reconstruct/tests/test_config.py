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
			global_heatmap_plotting:
			  default:
			    color_bar:
			      show_ticks: [1, 5, dynamic_high]
			      low_color: navy
			      mid_color: ivory
			      high_color: crimson
			      scale: log
			stages:
			  reconstruct:
			    execution:
			      force_restart: false
			    outputs:
			      output_rel_root: recon_outputs
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
	assert inputs.per_unit_outputs.write_gtr_pkl is True
	assert inputs.per_unit_outputs.write_gtr_json is True
	assert inputs.per_unit_outputs.write_amplitude_map_png is True
	assert inputs.per_unit_outputs.amplitude_map_png_relpath == "maps/amplitude_map.png"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.colorbar_location == "bottomleft"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.show_ticks == (2, 4, "dynamic_high")
	assert inputs.per_unit_outputs.amplitude_map_heatmap.low_color == "navy"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.scale == "log"
	assert inputs.per_unit_outputs.amplitude_map_heatmap.linear_cap_rounding_step == 5.0
	assert inputs.unit_ids == [94]

