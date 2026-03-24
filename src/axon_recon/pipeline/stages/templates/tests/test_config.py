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


def test_load_templates_config_parses_template_plots_waveforms_and_circles(tmp_path: Path) -> None:
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
			        template_plots:
			          waveforms:
			            write_png: true
			            relpath: maps/template_waveforms
			            channel_scope: all_channels
			          circles:
			            write_png: true
			            write_svg: true
			            relpath: maps/template_circles
			            channel_scope: recorded_channels
			            size_by: latency
			            color_by: amplitude
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template.relpath == "maps/template_waveforms"
	assert inputs.per_unit_outputs.template.channel_scope == "all_channels"
	assert inputs.per_unit_outputs.template_circles.write_png is True
	assert inputs.per_unit_outputs.template_circles.write_svg is True
	assert inputs.per_unit_outputs.template_circles.relpath == "maps/template_circles"
	assert inputs.per_unit_outputs.template_circles.channel_scope == "recorded_channels"
	assert inputs.per_unit_outputs.template_circles.size_by == "latency"
	assert inputs.per_unit_outputs.template_circles.color_by == "amplitude"


def test_load_templates_config_parses_template_plots_nested_under_full_template(tmp_path: Path) -> None:
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
			        full_template:
			          write_npy: false
			          template_plots:
			            waveforms:
			              write_png: true
			              relpath: nested/template_waveforms
			              channel_scope: all_channels
			            circles:
			              write_png: true
			              relpath: nested/template_circles
			              channel_scope: contributing_channels
			              size_by: amplitude
			              color_by: latency
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template.relpath == "nested/template_waveforms"
	assert inputs.per_unit_outputs.template.channel_scope == "all_channels"
	assert inputs.per_unit_outputs.template_circles.write_png is True
	assert inputs.per_unit_outputs.template_circles.relpath == "nested/template_circles"
	assert inputs.per_unit_outputs.template_circles.channel_scope == "contributing_channels"
	assert inputs.per_unit_outputs.template_circles.size_by == "amplitude"
	assert inputs.per_unit_outputs.template_circles.color_by == "latency"


def test_load_templates_config_parses_global_outputs_schema(tmp_path: Path) -> None:
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
			  outputs:
			    output_rel_root: template_outputs
			    per_unit_outputs:
			      full_template:
			        template_plots:
			          waveforms:
			            write_png: true
			            relpath: global/template_waveforms
			          circles:
			            write_png: true
			            relpath: global/template_circles
			            size_by: latency
			            color_by: amplitude
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.output_rel_root == "template_outputs"
	assert inputs.per_unit_outputs.template.relpath == "global/template_waveforms"
	assert inputs.per_unit_outputs.template_circles.write_png is True
	assert inputs.per_unit_outputs.template_circles.relpath == "global/template_circles"
	assert inputs.per_unit_outputs.template_circles.size_by == "latency"
	assert inputs.per_unit_outputs.template_circles.color_by == "amplitude"


def test_load_templates_config_parses_template_circles_color_bar_units(tmp_path: Path) -> None:
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
			  outputs:
			    per_unit_outputs:
			      full_template:
			        template_plots:
			          circles:
			            write_png: true
			            circle_size_scale_factor: 0.5
			            color_by: latency
			            color_bar:
			              units: ms
			              title: Latency (ms)
			              show_axes_title: false
			              show_unit_label: true
			              tick_decimal_places: 3
			              tick_target_count: 10
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template_circles.color_bar_units == "ms"
	assert inputs.per_unit_outputs.template_circles.circle_size_scale_factor == 0.5
	assert inputs.per_unit_outputs.template_circles.color_bar_title == "Latency (ms)"
	assert inputs.per_unit_outputs.template_circles.color_bar_show_axes_title is False
	assert inputs.per_unit_outputs.template_circles.color_bar_show_unit_labels is True
	assert inputs.per_unit_outputs.template_circles.color_bar_tick_decimal_places == 3
	assert inputs.per_unit_outputs.template_circles.color_bar_tick_target_count == 10


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


def test_load_templates_config_applies_global_heatmap_defaults(tmp_path: Path) -> None:
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
			global_heatmap_defaults:
			  default:
			    background: white
			    color_map: magma
			    color_bar:
			      show_color_bar: true
			      color_bar_fontsize: 11
			  footprint_plots:
			    default:
			      color_bar:
			        show: false
			        scale: log
			    amplitude_map:
			      relpath: global/amp
			  template_plots:
			    circles:
			      color_bar:
			        units: us
			        tick_decimal_places: 2
			stages:
			  templates:
			    outputs:
			      per_unit_outputs:
			        footprint_plots:
			          amplitude_map:
			            color_map: viridis
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	amp = inputs.per_unit_outputs.footprint_plots.amplitude_map
	assert amp.background == "white"
	assert amp.relpath == "global/amp"
	assert amp.color_map == "viridis"
	assert amp.show_color_bar is False
	assert amp.color_bar_fontsize == 11
	assert amp.scale == "log"

	circles = inputs.per_unit_outputs.template_circles
	assert circles.background == "white"
	assert circles.color_bar_units == "us"
	assert circles.color_bar_tick_decimal_places == 2


def test_load_templates_config_stage_values_override_global_heatmap_defaults(tmp_path: Path) -> None:
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
			global_heatmap_defaults:
			  circles:
			    color_bar:
			      units: us
			      tick_target_count: 7
			stages:
			  templates:
			    outputs:
			      per_unit_outputs:
			        template_plots:
			          circles:
			            color_bar:
			              units: ms
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	circles = inputs.per_unit_outputs.template_circles
	assert circles.color_bar_units == "ms"
	assert circles.color_bar_tick_target_count == 7


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
			          show_title: false
			          title_template: "Panel {{start}}-{{end}} of {{total}}"
			          title_fontsize: 11
			          top_channels: 30
			          channels_per_panel: 12
			          channel_overlap: 3
			          background: black
			          show_electrode_ids: true
			          channel_label_fontsize: 8
			          channel_label_x_offset_frac: 0.03
			          channel_label_y_offset_frac: 0.2
			          channel_label_alignment: right
			          trace_gain: 1.5
			          trace_spacing: 1.3
			          peak_marker_height_frac: 0.35
			          peak_marker_linewidth: 2.2
			          show_scale_bar: true
			          scale_bar_anchor_x_frac: 0.85
			          scale_bar_anchor_y_frac: 0.18
			          scale_bar_time_fraction: 0.2
			          scale_bar_amp_fraction: 0.3
			          force_amp_frac_to_max_amp: true
			          debug_max_amps_at_each_channel: true
			          bold_max_amp_channel_label: true
			          scale_bar_linewidth: 2.4
			          scale_bar_fontsize: 9
			          scale_bar_time_label_offset_frac: 0.05
			          scale_bar_amp_label_offset_frac: 0.04
			          latency_map:
			            show: true
			            color_map: cividis
			            force_square_aspect: false
			            title: Latency map test
			            fontsize: 9
			            template: full
			            color_bar:
			              show_color_bar: false
			              color_bar_fontsize: 8
			              color_bar_length_fraction: 0.25
			              color_bar_pad_fraction: 0.03
			              force_low_value: 2
			              force_high_value: 30
			              scale: linear
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
	assert prop.show_title is False
	assert prop.title_template == "Panel {start}-{end} of {total}"
	assert prop.title_fontsize == 11
	assert prop.top_channels == 30
	assert prop.channels_per_panel == 12
	assert prop.channel_overlap == 3
	assert prop.background == "black"
	assert prop.show_electrode_ids is True
	assert prop.channel_label_fontsize == 8
	assert prop.channel_label_x_offset_frac == 0.03
	assert prop.channel_label_y_offset_frac == 0.2
	assert prop.channel_label_alignment == "right"
	assert prop.trace_gain == 1.5
	assert prop.trace_spacing == 1.3
	assert prop.peak_marker_height_frac == 0.35
	assert prop.peak_marker_linewidth == 2.2
	assert prop.show_scale_bar is True
	assert prop.scale_bar_anchor_x_frac == 0.85
	assert prop.scale_bar_anchor_y_frac == 0.18
	assert prop.scale_bar_time_fraction == 0.2
	assert prop.scale_bar_amp_fraction == 0.3
	assert prop.force_amp_frac_to_max_amp is True
	assert prop.debug_max_amps_at_each_channel is True
	assert prop.bold_max_amp_channel_label is True
	assert prop.scale_bar_linewidth == 2.4
	assert prop.scale_bar_fontsize == 9
	assert prop.scale_bar_time_label_offset_frac == 0.05
	assert prop.scale_bar_amp_label_offset_frac == 0.04
	assert prop.latency_map.show is True
	assert prop.latency_map.color_map == "cividis"
	assert prop.latency_map.force_square_aspect is False
	assert prop.latency_map.title == "Latency map test"
	assert prop.latency_map.fontsize == 9
	assert prop.latency_map.template_shape == "full"
	assert prop.latency_map.show_color_bar is False
	assert prop.latency_map.color_bar_fontsize == 8
	assert prop.latency_map.color_bar_length_fraction == 0.25
	assert prop.latency_map.color_bar_pad_fraction == 0.03
	assert prop.latency_map.force_low_value == 2
	assert prop.latency_map.force_high_value == 30
	assert prop.latency_map.scale == "linear"


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
			          channel_locations_npy_relpath: arrays/merged_locs.npy
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
			          channel_locations_npy_relpath: arrays/full_locs.npy
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
	assert inputs.merge.max_waveforms_per_source_channel == 123
	assert inputs.merge.overlap_match_priority == ("electrode_id", "channel_id", "location")
	assert inputs.merge.location_tolerance_um == 2.5

	assert inputs.per_unit_outputs.merged_template.write_npy is True
	assert inputs.per_unit_outputs.merged_template.npy_relpath == "arrays/merged.npy"
	assert inputs.per_unit_outputs.merged_template.channel_locations_npy_relpath == "arrays/merged_locs.npy"
	assert inputs.per_unit_outputs.square_template.write_npy is True
	assert inputs.per_unit_outputs.square_template.padding_value == "nan"
	assert inputs.per_unit_outputs.square_template.channel_locations_npy_relpath == "square_channel_locations.npy"
	assert inputs.per_unit_outputs.scan_template.write_npy is True
	assert inputs.per_unit_outputs.scan_template.padding_value == "one"
	assert inputs.per_unit_outputs.scan_template.channel_locations_npy_relpath == "scan_channel_locations.npy"
	assert inputs.per_unit_outputs.full_template.write_npy is True
	assert inputs.per_unit_outputs.full_template.channel_locations_npy_relpath == "arrays/full_locs.npy"
	assert inputs.per_unit_outputs.full_template.padding_value == "zero"


def test_load_templates_config_parses_execution_upsampling_block(tmp_path: Path) -> None:
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
			      upsampling:
			        enable: true
			        method: sinc
			        factor: 10
			        mismatch_tolerance_hz: 0.25
			        raw_rate_fallback_hz: 10000
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.execution_upsampling.enabled is True
	assert inputs.execution_upsampling.factor == 10
	assert inputs.execution_upsampling.method == "sinc"
	assert inputs.execution_upsampling.mismatch_tolerance_hz == 0.25
	assert inputs.execution_upsampling.raw_rate_fallback_hz == 10000


def test_load_templates_config_parses_nested_alias_keys_from_debug_runtime(tmp_path: Path) -> None:
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
			        max_waveforms_per_source_channel: -1
			    outputs:
			      per_unit_outputs:
			        footprint_plots:
			          amplitude_map:
			            color_bar:
			              show_color_bar: false
			              color_bar_location: bottomright
			              color_bar_fontsize: 9
			              force_low_value: 2
			              scale: log
			            template:
			              shape: full
			          latency_map:
			            template:
			              shape: scan
			        propagation_plots:
			          latency_map:
			            template: full
			      reports:
			        footprint_grids:
			          amplitude_map_grid:
			            template: full
			          latency_map_grid:
			            template: scan
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	assert inputs.merge.max_waveforms_per_source_channel is None

	amp = inputs.per_unit_outputs.footprint_plots.amplitude_map
	assert amp.show_color_bar is False
	assert amp.color_bar_location == "bottomright"
	assert amp.color_bar_fontsize == 9
	assert amp.force_low_value == 2
	assert amp.scale == "log"
	assert amp.template_shape == "full"

	lat = inputs.per_unit_outputs.footprint_plots.latency_map
	assert lat.template_shape == "scan"

	assert inputs.per_unit_outputs.propagation_plots.latency_map.template_shape == "full"

	assert inputs.reports.footprint_grids.amplitude_map_grid.template_shape == "full"
	assert inputs.reports.footprint_grids.latency_map_grid.template_shape == "scan"


def test_load_templates_inputs_includes_probe_geometry_from_data_config(tmp_path: Path) -> None:
	data_path = tmp_path / "data.yml"
	data_path.write_text(
		dedent(
			"""
			output_root: /tmp/out
			Probe:
			  pitch_um: 17.5
			  electrode_size_um:
			    x: 12.0
			    y: 8.8
			  active_sensing_area_mm:
			    x: 3.85
			    y: 2.10
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
			  templates: {{}}
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.probe_geometry is not None
	assert inputs.probe_geometry.pitch_um == 17.5
	assert inputs.probe_geometry.electrode_size_um_x == 12.0
	assert inputs.probe_geometry.electrode_size_um_y == 8.8
	assert inputs.probe_geometry.active_area_um_x == 3850.0
	assert inputs.probe_geometry.active_area_um_y == 2100.0
