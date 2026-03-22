from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.pipeline.stages.templates.config import load_templates_inputs_from_runtime


def test_load_templates_config_from_templates_stage_block(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			datasets:
			  - raw_data_h5_path: /tmp/input.raw.h5
			    include_in_runtime: true
			    wells:
			      - well_id: well003
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
			    outputs:
			      output_rel_root: templates_stage_outputs
			      per_unit_outputs:
			        unit_reldir: units/{{unit_id:04d}}/
			        template:
			          write_png: true
			          write_svg: true
			          relpath: panel/template_view
			          channel_scope: all_channels
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path), unit_id_override=94)
	assert inputs.stream_id == "well003"
	assert inputs.output_rel_root == "templates_stage_outputs"
	assert inputs.per_unit_outputs.template.write_png is True
	assert inputs.per_unit_outputs.template.write_svg is True
	assert inputs.per_unit_outputs.template.relpath == "panel/template_view"
	assert inputs.per_unit_outputs.template.channel_scope == "all_channels"
	assert inputs.unit_ids == [94]


def test_load_templates_config_legacy_template_block(tmp_path: Path) -> None:
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
			        unit_reldir: units/{{unit_id:04d}}/
			        template:
			          write_png: false
			          write_svg: true
			          relpath: template_legacy
			          channel_scope: recorded_channels
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.stream_id == "well000"
	assert inputs.output_rel_root == "templates_outputs"
	assert inputs.per_unit_outputs.template.write_png is False
	assert inputs.per_unit_outputs.template.write_svg is True
	assert inputs.per_unit_outputs.template.relpath == "template_legacy"
	assert inputs.per_unit_outputs.template.channel_scope == "recorded_channels"


def test_load_templates_config_accepts_template_plot_alias(tmp_path: Path) -> None:
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
			    outputs:
			      per_unit_outputs:
			        template_plot:
			          write_png: false
			          write_svg: true
			          relpath: aliased/template_plot
			          channel_scope: recorded_channels
			          background: white
			          signal_color: black
			          force_center_soma: true
			          force_square_aspect: false
			          show_scale_bar: false
			          scale_bar_color: red
			          scale_bar_text_offset_frac: 0.03
			          scale_bar_y_offset_frac: 0.09
			          scale_bar_fontsize: 8
			          scale_bar_linewidth: 2.4
			          scale_bar_length_um: 75
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	tpl = inputs.per_unit_outputs.template
	assert tpl.write_png is False
	assert tpl.write_svg is True
	assert tpl.relpath == "aliased/template_plot"
	assert tpl.channel_scope == "recorded_channels"
	assert tpl.background == "white"
	assert tpl.signal_color == "black"
	assert tpl.force_center_soma is True
	assert tpl.force_square_aspect is False
	assert tpl.show_scale_bar is False
	assert tpl.scale_bar_color == "red"
	assert tpl.scale_bar_text_offset_frac == 0.03
	assert tpl.scale_bar_y_offset_frac == 0.09
	assert tpl.scale_bar_fontsize == 8
	assert tpl.scale_bar_linewidth == 2.4
	assert tpl.scale_bar_length_um == 75


def test_load_templates_config_prefers_canonical_template_key_over_alias(tmp_path: Path) -> None:
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
			    outputs:
			      per_unit_outputs:
			        template:
			          relpath: canonical/path
			          write_png: true
			        template_plot:
			          relpath: aliased/path
			          write_png: false
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template.relpath == "canonical/path"
	assert inputs.per_unit_outputs.template.write_png is True


def test_load_templates_config_parses_wf_overlay_and_execution_knobs(tmp_path: Path) -> None:
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
			    execution:
			      force_restart: false
			      force_replot: true
			      force_replot_per_unit: true
			      require_curated_units: false
			      unit_limit: 7
			      spikeinterface:
			        template_extraction:
			          sources:
			            include_concat: false
			            include_segments: true
			    outputs:
			      output_rel_root: template_outputs
			      reports:
			        plot_multi_source_pdf: true
			        multi_source_pdf_relpath: reports/template_multi_source.pdf
			        replot_from_disk: true
			        wf_overlay_grid:
			          write_pdf: true
			          pdf_relpath: reports/grid.pdf
			          write_png: false
			          png_relpath: reports/grid.png
			          top_channels_per_template: 15
			          time_upsample:
			            factor: 3
			            method: linear
			        footprint_grids:
			          amplitude_map_grid:
			            write_pdf: false
			            pdf_relpath: reports/amplitude_map_grid.pdf
			            write_png: true
			            png_relpath: reports/amplitude_map_grid.png
			          latency_map_grid:
			            write_pdf: true
			            pdf_relpath: reports/latency_map_grid.pdf
			            write_png: false
			            png_relpath: reports/latency_map_grid.png
			      per_unit_outputs:
			        template_wf_overlay:
			          write_pdf: true
			          pdf_relpath: reports/wf_overlay.pdf
			          write_png: true
			          png_relpath: reports/wf_overlay.png
			          top_channels_per_template: 12
			          include_mean: false
			          include_scale_bar: true
			          scale_bar_color: red
			          scale_bar_fontsize: 9
			          scale_bar_linewidth: 2.2
			          background: black
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.force_replot is True
	assert inputs.force_replot_per_unit is True
	assert inputs.require_curated_units is False
	assert inputs.unit_limit == 7
	assert inputs.include_concat is False
	assert inputs.include_segments is True

	overlay = inputs.per_unit_outputs.template_wf_overlay
	assert overlay.write_pdf is True
	assert overlay.pdf_relpath == "reports/wf_overlay.pdf"
	assert overlay.write_png is True
	assert overlay.png_relpath == "reports/wf_overlay.png"
	assert overlay.top_channels_per_template == 12
	assert overlay.include_mean is False
	assert overlay.include_scale_bar is True
	assert overlay.scale_bar_color == "red"
	assert overlay.scale_bar_fontsize == 9
	assert overlay.scale_bar_linewidth == 2.2
	assert overlay.background == "black"

	grid = inputs.reports.wf_overlay_grid
	assert grid.write_pdf is True
	assert grid.pdf_relpath == "reports/grid.pdf"
	assert grid.write_png is False
	assert grid.png_relpath == "reports/grid.png"
	assert grid.top_channels_per_template == 15

	assert inputs.reports.plot_multi_source_pdf.enabled is True
	assert inputs.reports.plot_multi_source_pdf.pdf_relpath == "reports/template_multi_source.pdf"
	assert inputs.reports.replot_from_disk is True
	assert inputs.reports.time_upsample.enabled is True
	assert inputs.reports.time_upsample.factor == 3
	assert inputs.reports.time_upsample.method == "linear"

	amp_grid = inputs.reports.footprint_grids.amplitude_map_grid
	assert amp_grid.write_pdf is False
	assert amp_grid.pdf_relpath == "reports/amplitude_map_grid.pdf"
	assert amp_grid.write_png is True
	assert amp_grid.png_relpath == "reports/amplitude_map_grid.png"

	lat_grid = inputs.reports.footprint_grids.latency_map_grid
	assert lat_grid.write_pdf is True
	assert lat_grid.pdf_relpath == "reports/latency_map_grid.pdf"
	assert lat_grid.write_png is False
	assert lat_grid.png_relpath == "reports/latency_map_grid.png"


def test_load_templates_config_accepts_foot_print_grids_alias(tmp_path: Path) -> None:
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
			    outputs:
			      reports:
			        foot_print_grids:
			          amplitude_map_grid:
			            write_pdf: true
			            pdf_relpath: old_alias_amp.pdf
			            write_png: false
			            png_relpath: old_alias_amp.png
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.reports.footprint_grids.amplitude_map_grid.write_pdf is True
	assert inputs.reports.footprint_grids.amplitude_map_grid.pdf_relpath == "old_alias_amp.pdf"
	assert inputs.reports.footprint_grids.amplitude_map_grid.write_png is False


def test_load_templates_config_parses_footprint_map_knobs(tmp_path: Path) -> None:
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
			    outputs:
			      per_unit_outputs:
			        footprint_plots:
			          amplitude_map:
			            write_png: true
			            write_svg: true
			            relpath: maps/amp
			            background: white
			            color_map: plasma
			            show_color_bar: false
			            color_bar_fontsize: 8
			            force_low_value: 1
			            force_high_value: 99
			            scale: log
			            show_ticks: [1, 10, dynamic_high]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	amp = inputs.per_unit_outputs.footprint_plots.amplitude_map
	assert amp.write_png is True
	assert amp.write_svg is True
	assert amp.relpath == "maps/amp"
	assert amp.background == "white"
	assert amp.color_map == "plasma"
	assert amp.show_color_bar is False
	assert amp.color_bar_fontsize == 8
	assert amp.force_low_value == 1
	assert amp.force_high_value == 99
	assert amp.scale == "log"
	assert amp.show_ticks == (1, 10, "dynamic_high")


def test_load_templates_config_time_upsample_defaults_to_sinc(tmp_path: Path) -> None:
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
			    outputs:
			      reports:
			        time_upsample:
			          enabled: true
			          factor: 2
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.reports.time_upsample.enabled is True
	assert inputs.reports.time_upsample.factor == 2
	assert inputs.reports.time_upsample.method == "sinc"


def test_load_templates_config_parses_topographical_and_propagation_blocks(tmp_path: Path) -> None:
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
			    outputs:
			      per_unit_outputs:
			        topographical_footprints:
			          amplitude:
			            write_png: true
			            write_svg: true
			            relpath: maps/topo_amp
			            color_map: plasma
			            background: white
			            elevation_deg: 20
			            azimuth_deg: -45
			            marker_size: 22
			          latency:
			            write_png: false
			            write_svg: true
			            relpath: maps/topo_lat
			            color_map: cividis
			            show_color_bar: false
			        propagation_plots:
			          write_pdf: true
			          pdf_relpath: maps/propagation.pdf
			          write_png: true
			          png_relpath: maps/propagation.png
			          top_channels: 30
			          channels_per_panel: 12
			          channel_overlap: 3
			          background: black
			          show_electrode_ids: true
			          trace_gain: 1.5
			          trace_spacing: 1.3
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	topo_amp = inputs.per_unit_outputs.topographical_footprints.amplitude
	assert topo_amp.write_png is True
	assert topo_amp.write_svg is True
	assert topo_amp.relpath == "maps/topo_amp"
	assert topo_amp.color_map == "plasma"
	assert topo_amp.background == "white"
	assert topo_amp.elevation_deg == 20
	assert topo_amp.azimuth_deg == -45
	assert topo_amp.marker_size == 22

	topo_lat = inputs.per_unit_outputs.topographical_footprints.latency
	assert topo_lat.write_png is False
	assert topo_lat.write_svg is True
	assert topo_lat.relpath == "maps/topo_lat"
	assert topo_lat.color_map == "cividis"
	assert topo_lat.show_color_bar is False

	prop = inputs.per_unit_outputs.propagation_plots
	assert prop.write_pdf is True
	assert prop.pdf_relpath == "maps/propagation.pdf"
	assert prop.write_png is True
	assert prop.png_relpath == "maps/propagation.png"
	assert prop.top_channels == 30
	assert prop.channels_per_panel == 12
	assert prop.channel_overlap == 3
	assert prop.background == "black"
	assert prop.show_electrode_ids is True
	assert prop.trace_gain == 1.5
	assert prop.trace_spacing == 1.3


def test_load_templates_config_parses_merge_and_template_artifact_knobs(tmp_path: Path) -> None:
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
			    execution:
			      merge:
			        enable: true
			        method: weighted_average
			        centering_method: pre_peak_robust_baseline
			        weighting_mode: per_channel_waveform_count
			        max_waveforms_per_source_channel: 123
			        overlap_match_priority: [electrode_id, channel_id, location]
			        location_tolerance_um: 2.5
			    outputs:
			      per_unit_outputs:
			        merged_template:
			          write_npy: true
			          npy_relpath: arrays/merged.npy
			        square_template:
			          write_npy: true
			          npy_relpath: arrays/square.npy
			          padding_value: nan
			        scan_template:
			          write_npy: true
			          npy_relpath: arrays/scan.npy
			          padding_value: one
			        full_template:
			          write_npy: true
			          npy_relpath: arrays/full.npy
			          padding_value: zero
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	assert inputs.merge.enable is True
	assert inputs.merge.method == "weighted_average"
	assert inputs.merge.centering_method == "pre_peak_robust_baseline"
	assert inputs.merge.weighting_mode == "per_channel_waveform_count"
	assert inputs.merge.max_waveforms_per_source_channel == 123
	assert inputs.merge.overlap_match_priority == ("electrode_id", "channel_id", "location")
	assert inputs.merge.location_tolerance_um == 2.5

	assert inputs.per_unit_outputs.merged_template.write_npy is True
	assert inputs.per_unit_outputs.merged_template.npy_relpath == "arrays/merged.npy"
	assert inputs.per_unit_outputs.square_template.write_npy is True
	assert inputs.per_unit_outputs.square_template.padding_value == "nan"
	assert inputs.per_unit_outputs.scan_template.write_npy is True
	assert inputs.per_unit_outputs.scan_template.padding_value == "one"
	assert inputs.per_unit_outputs.full_template.write_npy is True
	assert inputs.per_unit_outputs.full_template.padding_value == "zero"
