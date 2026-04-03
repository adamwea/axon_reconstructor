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


def test_load_templates_config_reads_runtime_unit_ids(tmp_path: Path) -> None:
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
			      unit_ids: [10, 17, 20, 10]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.unit_ids == [10, 17, 20]


def test_load_templates_config_prefers_unit_ids_override(tmp_path: Path) -> None:
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
			      unit_ids: [10, 17, 20]
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(
		config_path=str(runtime_path),
		unit_ids_override=[23, 24, 23],
	)
	assert inputs.unit_ids == [23, 24]


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
			            dpi: 360
			            relpath: maps/template_waveforms
			            channel_scope: all_channels
			            show_axes: false
			            unit_id_label:
			              show: true
			              fontsize: 14
			            center_most_channel_coords:
			              show: true
			              fontsize: 9
			          circles:
			            write_png: true
			            write_svg: true
			            dpi: 420
			            relpath: maps/template_circles
			            channel_scope: recorded_channels
			            size_by: latency
			            color_by: amplitude
			            show_axes: false
			            unit_id_label:
			              show: true
			              fontsize: 13
			            center_most_channel_coords:
			              show: true
			              fontsize: 8
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template.relpath == "maps/template_waveforms"
	assert inputs.per_unit_outputs.template.dpi == 360
	assert inputs.per_unit_outputs.template.channel_scope == "all_channels"
	assert inputs.per_unit_outputs.template.show_axes is False
	assert inputs.per_unit_outputs.template.unit_id_label.show is True
	assert inputs.per_unit_outputs.template.unit_id_label.fontsize == 14
	assert inputs.per_unit_outputs.template.center_most_channel_coords.show is True
	assert inputs.per_unit_outputs.template.center_most_channel_coords.fontsize == 9
	assert inputs.per_unit_outputs.template_circles.write_png is True
	assert inputs.per_unit_outputs.template_circles.write_svg is True
	assert inputs.per_unit_outputs.template_circles.dpi == 420
	assert inputs.per_unit_outputs.template_circles.relpath == "maps/template_circles"
	assert inputs.per_unit_outputs.template_circles.channel_scope == "recorded_channels"
	assert inputs.per_unit_outputs.template_circles.size_by == "latency"
	assert inputs.per_unit_outputs.template_circles.color_by == "amplitude"
	assert inputs.per_unit_outputs.template_circles.show_axes is False
	assert inputs.per_unit_outputs.template_circles.unit_id_label.show is True
	assert inputs.per_unit_outputs.template_circles.unit_id_label.fontsize == 13
	assert inputs.per_unit_outputs.template_circles.center_most_channel_coords.show is True
	assert inputs.per_unit_outputs.template_circles.center_most_channel_coords.fontsize == 8


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
			            color_by: latency
			            display:
			              scale_bar:
			                x_offset_frac: 0.25
			                x_offset_considers_fontsize: true
			                horizontal_alignment: left
			                vertical_alignment: top
			              show_scale_circle: true
			              scale_circle:
			                diameter: equal_to_max_amplitude
			                linewidth: 2.2
			                linestyle: solid
			                fontsize: 9
			                digits_after_decimal: 1
			                horizontal_alignment: left
			                vertical_alignment: top
			                x_offset_frac: 0.05
			                y_offset_frac: 0.07
			                font_location: inside
			                font_location_circle_too_small: below
			                units: uV
			              branch_morphology:
			                enabled: true
			                node_border_linewidth: 0.22
			                edge_linewidth: 1.1
			                show_branch_labels: true
			                unique_color_per_branch: true
			                color_scheme: tab10
			            color_bar:
			              units: ms
			              title: Latency (ms)
			              show_axes_title: false
			              show_unit_label: true
			              force_zero_and_neg_values_first_color_range: true
			              zero_transition_contrast: 2.0
			              tick_fontsize: 18
			              tick_decimal_places: 3
			              tick_target_count: 10
			            overlap_controls:
			              scalebar_coords_overlap_detect: true
			              scalebar_colorbar_overlap_detect: true
			              unitid_label_channel_overlap_detect: true
			              coords_channel_overlap_detect: true
			              scalebar_channel_overlap_detect: true
			              scalecircle_channel_overlap_detect: true
			              max_overlap_check_iterations: 5
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.per_unit_outputs.template_circles.color_bar_units == "ms"
	assert inputs.per_unit_outputs.template_circles.color_bar_title == "Latency (ms)"
	assert inputs.per_unit_outputs.template_circles.color_bar_show_axes_title is False
	assert inputs.per_unit_outputs.template_circles.color_bar_show_unit_labels is True
	assert inputs.per_unit_outputs.template_circles.color_bar_force_zero_and_neg_values_first_color_range is True
	assert inputs.per_unit_outputs.template_circles.color_bar_zero_transition_contrast == 2.0
	assert inputs.per_unit_outputs.template_circles.scale_bar_x_offset_frac == 0.25
	assert inputs.per_unit_outputs.template_circles.scale_bar_x_offset_considers_fontsize is True
	assert inputs.per_unit_outputs.template_circles.scale_bar_horizontal_alignment == "left"
	assert inputs.per_unit_outputs.template_circles.scale_bar_vertical_alignment == "top"
	assert inputs.per_unit_outputs.template_circles.show_scale_circle is True
	assert inputs.per_unit_outputs.template_circles.scale_circle_color == "white"
	assert inputs.per_unit_outputs.template_circles.scale_circle.diameter == "equal_to_max_amplitude"
	assert inputs.per_unit_outputs.template_circles.scale_circle.linewidth == 2.2
	assert inputs.per_unit_outputs.template_circles.scale_circle.linestyle == "solid"
	assert inputs.per_unit_outputs.template_circles.scale_circle.fontsize == 9
	assert inputs.per_unit_outputs.template_circles.scale_circle.digits_after_decimal == 1
	assert inputs.per_unit_outputs.template_circles.scale_circle.x_offset_frac == 0.05
	assert inputs.per_unit_outputs.template_circles.scale_circle.y_offset_frac == 0.07
	assert inputs.per_unit_outputs.template_circles.scale_circle.font_location == "inside"
	assert inputs.per_unit_outputs.template_circles.scale_circle.font_location_circle_too_small == "below"
	assert inputs.per_unit_outputs.template_circles.scale_circle.units == "uV"
	bm = inputs.per_unit_outputs.template_circles.branch_morphology
	assert bm.enabled is True
	assert bm.node_border_linewidth == 0.22
	assert bm.edge_linewidth == 1.1
	assert bm.show_branch_labels is True
	assert bm.unique_color_per_branch is True
	assert bm.color_scheme == "tab10"
	assert inputs.per_unit_outputs.template_circles.color_bar_tick_fontsize == 18
	assert inputs.per_unit_outputs.template_circles.color_bar_tick_decimal_places == 3
	assert inputs.per_unit_outputs.template_circles.color_bar_tick_target_count == 10
	overlap = inputs.per_unit_outputs.template_circles.overlap_controls
	assert overlap.scalebar_coords_overlap_detect is True
	assert overlap.scalebar_colorbar_overlap_detect is True
	assert overlap.unitid_label_channel_overlap_detect is True
	assert overlap.coords_channel_overlap_detect is True
	assert overlap.scalebar_channel_overlap_detect is True
	assert overlap.scalecircle_channel_overlap_detect is True
	assert overlap.max_overlap_check_iterations == 5


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
			      analyzer_cache:
			        enabled: true
			        relpath: cache/analyzers
			        cleanup_on_success: true
			        reuse_on_force_restart: true
			      reports:
			        plot_multi_source_pdf: true
			        multi_source_pdf_relpath: reports/template_multi_source.pdf
			        replot_from_disk: true
			        overwrite_on_unit_rerun: true
			        wf_overlay_grid:
			          write_pdf: true
			          pdf_relpath: reports/grid.pdf
			          write_png: false
			          png_relpath: reports/grid.png
			          write_svg: true
			          svg_relpath: reports/grid.svg
			          keep_temp_svg: true
			          temp_svg_relpath: reports/grid__temp.svg
			          top_channels_per_template: 15
			          subplot_background_color: black
			          figure_background_color: black
			          render_mode: image_composite
			          dpi: 360
			          show_title: true
			          show_axes: true
			          show_channel_labels: true
			          time_upsample:
			            factor: 3
			            method: linear
			        footprint_grids:
			          circles_map_grid:
			            show_title: false
			            write_pdf: true
			            pdf_relpath: reports/circles_map_grid.pdf
			            write_png: true
			            png_relpath: reports/circles_map_grid.png
			            write_svg: true
			            svg_relpath: reports/circles_map_grid.svg
			            keep_temp_svg: true
			            temp_svg_relpath: reports/circles_map_grid__temp.svg
			            render_mode: direct_replot
			            dpi: 420
			            subplot_background_color: black
			            figure_background_color: black
			          amplitude_map_grid:
			            show_title: false
			            write_pdf: false
			            pdf_relpath: reports/amplitude_map_grid.pdf
			            write_png: true
			            png_relpath: reports/amplitude_map_grid.png
			            write_svg: true
			            svg_relpath: reports/amplitude_map_grid.svg
			            keep_temp_svg: true
			            temp_svg_relpath: reports/amplitude_map_grid__temp.svg
			            render_mode: image_composite
			            dpi: 310
			            subplot_background_color: black
			            figure_background_color: white
			          latency_map_grid:
			            show_title: true
			            write_pdf: true
			            pdf_relpath: reports/latency_map_grid.pdf
			            write_png: false
			            png_relpath: reports/latency_map_grid.png
			            write_svg: true
			            svg_relpath: reports/latency_map_grid.svg
			            keep_temp_svg: false
			            temp_svg_relpath: reports/latency_map_grid__temp.svg
			            render_mode: direct_replot
			            dpi: 500
			            subplot_background_color: white
			            figure_background_color: black
			      per_unit_outputs:
			        template_wf_overlay:
			          debug_mode: true
			          write_pdf: true
			          pdf_relpath: reports/wf_overlay.pdf
			          write_png: true
			          png_relpath: reports/wf_overlay.png
			          top_channels_per_template: 12
			          show_title: false
			          show_axes: false
			          show_channel_labels: false
			          show_top_channel_info: false
			          show_waveform_count_info: false
			          include_mean: false
			          max_waveforms_to_show: 77
			          waveform_sampling_mode: random
			          random_seed: 123
			          include_scale_bar: true
			          scale_bar_color: red
			          scale_bar_fontsize: 9
			          scale_bar_linewidth: 2.2
			          scale_bar_time_fraction: 0.12
			          scale_bar_amp_fraction: 0.16
			          scale_bar_time_label_offset_frac: 0.05
			          scale_bar_amp_label_offset_frac: 0.03
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
	assert inputs.analyzer_cache.enabled is True
	assert inputs.analyzer_cache.relpath == "cache/analyzers"
	assert inputs.analyzer_cache.cleanup_on_success is True
	assert inputs.analyzer_cache.reuse_on_force_restart is True

	overlay = inputs.per_unit_outputs.template_wf_overlay
	assert overlay.debug_mode is True
	assert overlay.write_pdf is True
	assert overlay.pdf_relpath == "reports/wf_overlay.pdf"
	assert overlay.write_png is True
	assert overlay.png_relpath == "reports/wf_overlay.png"
	assert overlay.top_channels_per_template == 12
	assert overlay.show_title is False
	assert overlay.show_axes is False
	assert overlay.show_channel_labels is False
	assert overlay.show_top_channel_info is False
	assert overlay.show_waveform_count_info is False
	assert overlay.max_waveforms_to_show == 77
	assert overlay.waveform_sampling_mode == "random"
	assert overlay.random_seed == 123
	assert overlay.include_mean is False
	assert overlay.include_scale_bar is True
	assert overlay.scale_bar_color == "red"
	assert overlay.scale_bar_fontsize == 9
	assert overlay.scale_bar_linewidth == 2.2
	assert overlay.scale_bar_time_fraction == 0.12
	assert overlay.scale_bar_amp_fraction == 0.16
	assert overlay.scale_bar_time_label_offset_frac == 0.05
	assert overlay.scale_bar_amp_label_offset_frac == 0.03
	assert overlay.background == "black"

	grid = inputs.reports.wf_overlay_grid
	assert grid.write_pdf is True
	assert grid.pdf_relpath == "reports/grid.pdf"
	assert grid.write_png is False
	assert grid.png_relpath == "reports/grid.png"
	assert grid.write_svg is True
	assert grid.svg_relpath == "reports/grid.svg"
	assert grid.keep_temp_svg is True
	assert grid.temp_svg_relpath == "reports/grid__temp.svg"
	assert grid.top_channels_per_template == 15
	assert grid.subplot_background_color == "black"
	assert grid.figure_background_color == "black"
	assert grid.render_mode == "image_composite"
	assert grid.dpi == 360

	assert inputs.reports.plot_multi_source_pdf.enabled is True
	assert inputs.reports.plot_multi_source_pdf.pdf_relpath == "reports/template_multi_source.pdf"
	assert inputs.reports.replot_from_disk is True
	assert inputs.reports.overwrite_on_unit_rerun is True
	assert inputs.reports.time_upsample.enabled is True
	assert inputs.reports.time_upsample.factor == 3
	assert inputs.reports.time_upsample.method == "linear"

	circles_grid = inputs.reports.footprint_grids.circles_map_grid
	assert circles_grid.show_title is False
	assert circles_grid.write_pdf is True
	assert circles_grid.pdf_relpath == "reports/circles_map_grid.pdf"
	assert circles_grid.write_png is True
	assert circles_grid.png_relpath == "reports/circles_map_grid.png"
	assert circles_grid.write_svg is True
	assert circles_grid.svg_relpath == "reports/circles_map_grid.svg"
	assert circles_grid.keep_temp_svg is True
	assert circles_grid.temp_svg_relpath == "reports/circles_map_grid__temp.svg"
	assert circles_grid.render_mode == "direct_replot"
	assert circles_grid.dpi == 420
	assert circles_grid.subplot_background_color == "black"
	assert circles_grid.figure_background_color == "black"

	amp_grid = inputs.reports.footprint_grids.amplitude_map_grid
	assert amp_grid.show_title is False
	assert amp_grid.write_pdf is False
	assert amp_grid.pdf_relpath == "reports/amplitude_map_grid.pdf"
	assert amp_grid.write_png is True
	assert amp_grid.png_relpath == "reports/amplitude_map_grid.png"
	assert amp_grid.write_svg is True
	assert amp_grid.svg_relpath == "reports/amplitude_map_grid.svg"
	assert amp_grid.keep_temp_svg is True
	assert amp_grid.temp_svg_relpath == "reports/amplitude_map_grid__temp.svg"
	assert amp_grid.render_mode == "image_composite"
	assert amp_grid.dpi == 310
	assert amp_grid.subplot_background_color == "black"
	assert amp_grid.figure_background_color == "white"

	lat_grid = inputs.reports.footprint_grids.latency_map_grid
	assert lat_grid.show_title is True
	assert lat_grid.write_pdf is True
	assert lat_grid.pdf_relpath == "reports/latency_map_grid.pdf"
	assert lat_grid.write_png is False
	assert lat_grid.png_relpath == "reports/latency_map_grid.png"
	assert lat_grid.write_svg is True
	assert lat_grid.svg_relpath == "reports/latency_map_grid.svg"
	assert lat_grid.keep_temp_svg is False
	assert lat_grid.temp_svg_relpath == "reports/latency_map_grid__temp.svg"
	assert lat_grid.render_mode == "direct_replot"
	assert lat_grid.dpi == 500
	assert lat_grid.subplot_background_color == "white"
	assert lat_grid.figure_background_color == "black"


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


def test_load_templates_config_parses_nested_report_grid_blocks(tmp_path: Path) -> None:
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
			        grids:
			          wf_overlay_grid:
			            write_png: false
			            output:
			              write_png: true
			              png_relpath: nested/wf_overlay_grid.png
			              write_svg: true
			              svg_relpath: nested/wf_overlay_grid.svg
			              keep_temp_svg: true
			              temp_svg_relpath: nested/wf_overlay_grid__temp.svg
			            render:
			              mode: direct_replot
			              dpi: 410
			              subplot_background_color: black
			              figure_background_color: black
			            display:
			              top_channels_per_template: 14
			          footprint_grids:
			            circles_map_grid:
			              output:
			                write_png: true
			                png_relpath: nested/circles.png
			                write_svg: true
			                svg_relpath: nested/circles.svg
			                keep_temp_svg: true
			                temp_svg_relpath: nested/circles__temp.svg
			              display:
			                show_title: false
			              render:
			                mode: direct_replot
			                dpi: 430
			                template: square
			                global_color_scale: true
			                subplot_background_color: black
			                figure_background_color: black
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	wf_grid = inputs.reports.wf_overlay_grid
	assert wf_grid.write_png is True
	assert wf_grid.png_relpath == "nested/wf_overlay_grid.png"
	assert wf_grid.write_svg is True
	assert wf_grid.svg_relpath == "nested/wf_overlay_grid.svg"
	assert wf_grid.keep_temp_svg is True
	assert wf_grid.temp_svg_relpath == "nested/wf_overlay_grid__temp.svg"
	assert wf_grid.render_mode == "direct_replot"
	assert wf_grid.dpi == 410
	assert wf_grid.subplot_background_color == "black"
	assert wf_grid.figure_background_color == "black"
	assert wf_grid.top_channels_per_template == 14

	circles_grid = inputs.reports.footprint_grids.circles_map_grid
	assert circles_grid.write_png is True
	assert circles_grid.png_relpath == "nested/circles.png"
	assert circles_grid.write_svg is True
	assert circles_grid.svg_relpath == "nested/circles.svg"
	assert circles_grid.keep_temp_svg is True
	assert circles_grid.temp_svg_relpath == "nested/circles__temp.svg"
	assert circles_grid.show_title is False
	assert circles_grid.render_mode == "direct_replot"
	assert circles_grid.dpi == 430
	assert circles_grid.template_shape == "square"
	assert circles_grid.global_color_scale is True
	assert circles_grid.subplot_background_color == "black"
	assert circles_grid.figure_background_color == "black"


def test_load_templates_config_parses_nested_extremum_wf_overlay_blocks(tmp_path: Path) -> None:
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
			        extremum_ch_wf_overlay:
			          output:
			            write_pdf: true
			            pdf_relpath: nested/overlay.pdf
			            write_png: true
			            png_relpath: nested/overlay.png
			          display:
			            debug_mode: true
			            top_channels_per_template: 11
			            show_title: false
			            show_axes: false
			            show_channel_labels: false
			            show_top_channel_info: false
			            show_waveform_count_info: false
			            include_mean: false
			            max_waveforms_to_show: 55
			            waveform_sampling_mode: random
			            random_seed: 42
			          scale_bar:
			            include: true
			            color: red
			            fontsize: 9
			            linewidth: 2.0
			            time_fraction: 0.12
			            amp_fraction: 0.15
			            time_label_offset_frac: 0.04
			            amp_label_offset_frac: 0.03
			          render:
			            background: black
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	overlay = inputs.per_unit_outputs.template_wf_overlay

	assert overlay.write_pdf is True
	assert overlay.pdf_relpath == "nested/overlay.pdf"
	assert overlay.write_png is True
	assert overlay.png_relpath == "nested/overlay.png"
	assert overlay.debug_mode is True
	assert overlay.top_channels_per_template == 11
	assert overlay.show_title is False
	assert overlay.show_axes is False
	assert overlay.show_channel_labels is False
	assert overlay.show_top_channel_info is False
	assert overlay.show_waveform_count_info is False
	assert overlay.include_mean is False
	assert overlay.max_waveforms_to_show == 55
	assert overlay.waveform_sampling_mode == "random"
	assert overlay.random_seed == 42
	assert overlay.include_scale_bar is True
	assert overlay.scale_bar_color == "red"
	assert overlay.scale_bar_fontsize == 9
	assert overlay.scale_bar_linewidth == 2.0
	assert overlay.scale_bar_time_fraction == 0.12
	assert overlay.scale_bar_amp_fraction == 0.15
	assert overlay.scale_bar_time_label_offset_frac == 0.04
	assert overlay.scale_bar_amp_label_offset_frac == 0.03
	assert overlay.background == "black"


def test_load_templates_config_parses_nested_template_and_footprint_blocks(tmp_path: Path) -> None:
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
			            output:
			              write_png: true
			              write_svg: false
			              relpath: nested/template_waveforms
			            display:
			              channel_scope: all_channels
			              show_axes: false
			            render:
			              background: black
			              signal_color: white
			          circles:
			            output:
			              write_png: true
			              write_svg: true
			              dpi: 440
			              relpath: nested/template_circles
			            display:
			              channel_scope: recorded_channels
			              show_axes: false
			              size_by: latency
			              color_by: amplitude
			            render:
			              background: black
			              signal_color: white
			        footprint_plots:
			          amplitude_map:
			            output:
			              write_png: true
			              write_svg: false
			              relpath: nested/amp
			            render:
			              background: white
			              color_map: plasma
			          latency_map:
			            output:
			              write_png: true
			              write_svg: false
			              relpath: nested/lat
			        topographical_footprints:
			          amplitude:
			            output:
			              write_png: true
			              write_svg: false
			              relpath: nested/topo_amp
			            display:
			              elevation_deg: 22
			              azimuth_deg: -40
			              marker_size: 18
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	wf = inputs.per_unit_outputs.template
	assert wf.relpath == "nested/template_waveforms"
	assert wf.channel_scope == "all_channels"
	assert wf.show_axes is False
	assert wf.background == "black"
	assert wf.signal_color == "white"

	circles = inputs.per_unit_outputs.template_circles
	assert circles.relpath == "nested/template_circles"
	assert circles.dpi == 440
	assert circles.channel_scope == "recorded_channels"
	assert circles.show_axes is False
	assert circles.size_by == "latency"
	assert circles.color_by == "amplitude"

	amp = inputs.per_unit_outputs.footprint_plots.amplitude_map
	assert amp.relpath == "nested/amp"
	assert amp.background == "white"
	assert amp.color_map == "plasma"

	lat = inputs.per_unit_outputs.footprint_plots.latency_map
	assert lat.relpath == "nested/lat"

	topo = inputs.per_unit_outputs.topographical_footprints.amplitude
	assert topo.relpath == "nested/topo_amp"
	assert topo.elevation_deg == 22
	assert topo.azimuth_deg == -40
	assert topo.marker_size == 18


def test_load_templates_config_parses_template_scale_bar_under_display(tmp_path: Path) -> None:
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
			            display:
			              scale_bar:
			                text_offset_frac: 0.031
			                y_offset_frac: 0.071
			                fontsize: 11
			                linewidth: 2.3
			                length_um: 80
			          circles:
			            display:
			              scale_bar:
			                text_offset_frac: 0.017
			                y_offset_frac: 0.022
			                fontsize: 15
			                linewidth: 4.1
			                length_um: 55
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))

	wf = inputs.per_unit_outputs.template
	assert wf.scale_bar_text_offset_frac == 0.031
	assert wf.scale_bar_y_offset_frac == 0.071
	assert wf.scale_bar_fontsize == 11
	assert wf.scale_bar_linewidth == 2.3
	assert wf.scale_bar_length_um == 80

	circles = inputs.per_unit_outputs.template_circles
	assert circles.scale_bar_text_offset_frac == 0.017
	assert circles.scale_bar_y_offset_frac == 0.022
	assert circles.scale_bar_fontsize == 15
	assert circles.scale_bar_linewidth == 4.1
	assert circles.scale_bar_length_um == 55


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
			          write_svg: true
			          write_circles_template_numbered_png: true
			          write_circles_template_numbered_svg: false
			          circles_template_numbered_relpath: maps/circles_numbered
			          write_propagation_2panel_png: true
			          write_propagation_2panel_svg: true
			          propagation_2panel_relpath: maps/propagation_2panel
			          show_title: false
			          title_template: "Panel {{start}}-{{end}} of {{total}}"
			          title_fontsize: 11
			          top_channels: 30
			          window_strategy: first_k
			          channels_per_panel: 12
			          channel_overlap: 3
			          force_start_with_max_ptp: false
			          force_start_with_max_negative_peak: true
			          force_min_neg_peak_index_zero: true
			          ordering_latency_mode: negative_peak
			          latency_tie_breaker: channel_index
			          trace_label_mode: order_index
			          relative_signed_order_numbers: true
			          show_right_panel: true
			          right_panel_gap_fraction: 0.08
			          right_panel_width_scale: 0.85
			          right_panel_keep_temp_svg: true
			          right_panel_svg_relpath: maps/propagation_right.svg
			          right_panel_png_relpath: maps/propagation_right.png
			          left_panel_png_dpi: 550
			          right_panel_png_dpi: 450
			          composed_png_dpi: 700
			          background: black
			          show_electrode_ids: true
			          electrode_label_fontsize: 8
			          electrode_label_x_offset_frac: 0.03
			          electrode_label_y_offset_frac: 0.2
			          electrode_label_alignment: right
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
			          bold_max_amp_electrode_label: true
			          scale_bar_linewidth: 2.4
			          scale_bar_fontsize: 9
			          scale_bar_time_label_offset_frac: 0.05
			          scale_bar_amp_label_offset_frac: 0.04
			          post_ap_abbrev:
			            enabled: true
			            start_ms: 1.25
			            start_samples: 12
			            cut_fraction: 0.6
			            min_samples_to_cut: 8
			            gap_samples: 6
			            marker_text: /.../
			            marker_fontsize: 11
			            marker_y_offset_frac: 0.0
			          duration_info:
			            show: true
			            x_frac: 0.62
			            y_frac: 0.92
			            fontsize: 8
			            horizontal_alignment: right
			            vertical_alignment: top
			          plot_layout:
			            width_in: 12.0
			            panel_height_in: 2.2
			            extra_height_in: 0.8
			            hspace: 0.2
			            area_aspect_ratio: 6.0
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
	assert prop.write_svg is True
	assert prop.write_circles_template_numbered_png is True
	assert prop.write_circles_template_numbered_svg is False
	assert prop.circles_template_numbered_relpath == "maps/circles_numbered"
	assert prop.write_propagation_2panel_png is True
	assert prop.write_propagation_2panel_svg is True
	assert prop.propagation_2panel_relpath == "maps/propagation_2panel"
	assert prop.show_title is False
	assert prop.title_template == "Panel {start}-{end} of {total}"
	assert prop.title_fontsize == 11
	assert prop.top_channels == 30
	assert prop.window_strategy == "first_k"
	assert prop.channels_per_panel == 12
	assert prop.channel_overlap == 3
	assert prop.force_start_with_max_ptp is False
	assert prop.force_start_with_max_negative_peak is True
	assert prop.force_min_neg_peak_index_zero is True
	assert prop.ordering_latency_mode == "negative_peak"
	assert prop.latency_tie_breaker == "channel_index"
	assert prop.trace_label_mode == "order_index"
	assert prop.relative_signed_order_numbers is True
	assert prop.show_right_panel is True
	assert prop.right_panel_gap_fraction == 0.08
	assert prop.right_panel_width_scale == 0.85
	assert prop.right_panel_keep_temp_svg is True
	assert prop.right_panel_svg_relpath == "maps/propagation_right.svg"
	assert prop.right_panel_png_relpath == "maps/propagation_right.png"
	assert prop.left_panel_png_dpi == 550
	assert prop.right_panel_png_dpi == 450
	assert prop.composed_png_dpi == 700
	assert prop.background == "black"
	assert prop.show_electrode_ids is True
	assert prop.electrode_label_fontsize == 8
	assert prop.electrode_label_x_offset_frac == 0.03
	assert prop.electrode_label_y_offset_frac == 0.2
	assert prop.electrode_label_alignment == "right"
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
	assert prop.bold_max_amp_electrode_label is True
	assert prop.scale_bar_linewidth == 2.4
	assert prop.scale_bar_fontsize == 9
	assert prop.scale_bar_time_label_offset_frac == 0.05
	assert prop.scale_bar_amp_label_offset_frac == 0.04
	assert prop.abbreviate_post_ap_signal is True
	assert prop.post_ap_abbrev_start_ms == 1.25
	assert prop.post_ap_abbrev_start_samples == 12
	assert prop.post_ap_abbrev_cut_fraction == 0.6
	assert prop.post_ap_abbrev_min_samples_to_cut == 8
	assert prop.post_ap_abbrev_gap_samples == 6
	assert prop.post_ap_abbrev_marker_text == "/.../"
	assert prop.post_ap_abbrev_marker_fontsize == 11
	assert prop.post_ap_abbrev_marker_y_offset_frac == 0.0
	assert prop.show_duration_info is True
	assert prop.duration_info_x_frac == 0.62
	assert prop.duration_info_y_frac == 0.92
	assert prop.duration_info_fontsize == 8
	assert prop.duration_info_horizontal_alignment == "right"
	assert prop.duration_info_vertical_alignment == "top"
	assert prop.plot_width_in == 12.0
	assert prop.plot_panel_height_in == 2.2
	assert prop.plot_extra_height_in == 0.8
	assert prop.plot_hspace == 0.2
	assert prop.plot_area_aspect_ratio == 6.0
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


def test_load_templates_config_parses_nested_propagation_groups(tmp_path: Path) -> None:
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
			        propagation_plots:
			          output:
			            write_pdf: true
			            pdf_relpath: nested/propagation.pdf
			            write_png: true
			            png_relpath: nested/propagation.png
			            write_svg: false
			            write_circles_template_numbered_png: false
			            write_circles_template_numbered_svg: true
			            circles_template_numbered_relpath: nested/circles_numbered
			            write_propagation_2panel_png: false
			            write_propagation_2panel_svg: true
			            propagation_2panel_relpath: nested/propagation_2panel
			          display:
			            show_title: false
			            title_template: "Nested {{start}}-{{end}}"
			            title_fontsize: 10
			            top_channels: 28
			            channels_per_panel: 10
			            channel_overlap: 2
			            force_start_with_max_ptp: true
			            force_start_with_max_negative_peak: false
			            trace_label_mode: electrode_id
			            relative_signed_order_numbers: false
			            show_right_panel: false
			            right_panel_gap_fraction: 0.05
			            right_panel_width_scale: 1.2
			            right_panel_keep_temp_svg: false
			            right_panel_svg_relpath: nested/right_panel.svg
			            right_panel_png_relpath: nested/right_panel.png
			            left_panel_png_dpi: 500
			            right_panel_png_dpi: 500
			            composed_png_dpi: 800
			          render:
			            background: black
			            trace_gain: 1.7
			            trace_spacing: 1.2
			            peak_marker_height_frac: 0.40
			            peak_marker_linewidth: 0.8
			          labels:
			            show_electrode_ids: true
			            electrode_label_fontsize: 7
			            electrode_label_x_offset_frac: -0.01
			            electrode_label_y_offset_frac: 0.02
			            electrode_label_alignment: right
			            bold_max_amp_electrode_label: true
			          scale_bar:
			            show: true
			            anchor_x_frac: -0.12
			            anchor_y_frac: 0.11
			            time_fraction: 0.08
			            amp_fraction: 0.45
			            force_amp_frac_to_max_amp: true
			            debug_max_amps_at_each_channel: false
			            linewidth: 2.0
			            fontsize: 8
			            time_label_offset_frac: 0.03
			            amp_label_offset_frac: 0.02
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	prop = inputs.per_unit_outputs.propagation_plots

	assert prop.write_pdf is True
	assert prop.pdf_relpath == "nested/propagation.pdf"
	assert prop.write_png is True
	assert prop.png_relpath == "nested/propagation.png"
	assert prop.write_svg is False
	assert prop.write_circles_template_numbered_png is False
	assert prop.write_circles_template_numbered_svg is True
	assert prop.circles_template_numbered_relpath == "nested/circles_numbered"
	assert prop.write_propagation_2panel_png is False
	assert prop.write_propagation_2panel_svg is True
	assert prop.propagation_2panel_relpath == "nested/propagation_2panel"
	assert prop.show_title is False
	assert prop.title_template == "Nested {start}-{end}"
	assert prop.title_fontsize == 10
	assert prop.top_channels == 28
	assert prop.channels_per_panel == 10
	assert prop.channel_overlap == 2
	assert prop.force_start_with_max_ptp is True
	assert prop.force_start_with_max_negative_peak is False
	assert prop.trace_label_mode == "electrode_id"
	assert prop.relative_signed_order_numbers is False
	assert prop.show_right_panel is False
	assert prop.right_panel_gap_fraction == 0.05
	assert prop.right_panel_width_scale == 1.2
	assert prop.right_panel_keep_temp_svg is False
	assert prop.right_panel_svg_relpath == "nested/right_panel.svg"
	assert prop.right_panel_png_relpath == "nested/right_panel.png"
	assert prop.left_panel_png_dpi == 500
	assert prop.right_panel_png_dpi == 500
	assert prop.composed_png_dpi == 800
	assert prop.background == "black"
	assert prop.trace_gain == 1.7
	assert prop.trace_spacing == 1.2
	assert prop.peak_marker_height_frac == 0.40
	assert prop.peak_marker_linewidth == 0.8
	assert prop.show_electrode_ids is True
	assert prop.electrode_label_fontsize == 7
	assert prop.electrode_label_x_offset_frac == -0.01
	assert prop.electrode_label_y_offset_frac == 0.02
	assert prop.electrode_label_alignment == "right"
	assert prop.bold_max_amp_electrode_label is True
	assert prop.show_scale_bar is True
	assert prop.scale_bar_anchor_x_frac == -0.12
	assert prop.scale_bar_anchor_y_frac == 0.11
	assert prop.scale_bar_time_fraction == 0.08
	assert prop.scale_bar_amp_fraction == 0.45
	assert prop.force_amp_frac_to_max_amp is True
	assert prop.debug_max_amps_at_each_channel is False
	assert prop.scale_bar_linewidth == 2.0
	assert prop.scale_bar_fontsize == 8
	assert prop.scale_bar_time_label_offset_frac == 0.03
	assert prop.scale_bar_amp_label_offset_frac == 0.02


def test_load_templates_config_parses_nested_split_propagation_output_blocks(tmp_path: Path) -> None:
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
			        propagation_plots:
			          output:
			            propagation_plot:
			              write_pdf: true
			              pdf_relpath: clean/propagation.pdf
			              write_png: true
			              png_relpath: clean/propagation.png
			              write_svg: false
			              png_dpi: 510
			            circles_template_numbered:
			              write_png: false
			              write_svg: true
			              relpath: clean/circles_numbered
			              png_dpi: 520
			            propagation_2panel:
			              write_png: false
			              write_svg: true
			              relpath: clean/propagation_2panel
			              png_dpi: 530
			              layout:
			                gap_fraction: 0.01
			                width_scale: 0.9
			                keep_temp_svg: true
			                right_panel_svg_relpath: clean/right_temp.svg
			                right_panel_png_relpath: clean/right_temp.png
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	prop = inputs.per_unit_outputs.propagation_plots

	assert prop.write_pdf is True
	assert prop.pdf_relpath == "clean/propagation.pdf"
	assert prop.write_png is True
	assert prop.png_relpath == "clean/propagation.png"
	assert prop.write_svg is False
	assert prop.write_circles_template_numbered_png is False
	assert prop.write_circles_template_numbered_svg is True
	assert prop.circles_template_numbered_relpath == "clean/circles_numbered"
	assert prop.write_propagation_2panel_png is False
	assert prop.write_propagation_2panel_svg is True
	assert prop.propagation_2panel_relpath == "clean/propagation_2panel"
	assert prop.left_panel_png_dpi == 510
	assert prop.right_panel_png_dpi == 520
	assert prop.composed_png_dpi == 530
	assert prop.right_panel_gap_fraction == 0.01
	assert prop.right_panel_width_scale == 0.9
	assert prop.right_panel_keep_temp_svg is True
	assert prop.right_panel_svg_relpath == "clean/right_temp.svg"
	assert prop.right_panel_png_relpath == "clean/right_temp.png"


def test_load_templates_config_parses_template_circles_propagation_order_labels(tmp_path: Path) -> None:
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
			          circles:
			            output:
			              write_png: true
			            propagation_order_labels:
			              show: true
			              fontsize: 7
			              color: yellow
			              bbox_alpha: 0.2
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	circles = inputs.per_unit_outputs.template_circles
	assert circles.show_propagation_order_labels is True
	assert circles.propagation_order_label_fontsize == 7
	assert circles.propagation_order_label_color == "yellow"
	assert circles.propagation_order_label_bbox_alpha == 0.2


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


def test_load_templates_config_parses_waveform_extraction_controls_with_fallback(tmp_path: Path) -> None:
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
			  waveforms:
			    ms_before: 1.1
			    ms_after: 2.2
			    max_spikes_per_unit: 999
			  templates:
			    execution:
			      inputs:
			        preprocessed_segments_reldir: /preprocess_outputs/per_segment_recordings
			        preprocessed_concat_reldir: /preprocess_outputs/preprocessed_recording
			        concat_sorting_relpath: /spikesort_outputs/sorter_output
			        concat_analyzer_relpath: /stg2_spikesorting_outputs/analyzer_output
			      spikeinterface:
			        waveform_extraction:
			          window:
			            ms_before: 1.5
			            ms_after: 2.5
			          max_spikes_per_unit: -1
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.waveform_extraction.ms_before == 1.5
	assert inputs.waveform_extraction.ms_after == 2.5
	assert inputs.waveform_extraction.max_spikes_per_unit is None
	assert inputs.preprocessed_segments_reldir == "/preprocess_outputs/per_segment_recordings"
	assert inputs.preprocessed_concat_reldir == "/preprocess_outputs/preprocessed_recording"
	assert inputs.concat_sorting_relpath == "/spikesort_outputs/sorter_output"
	assert inputs.concat_analyzer_relpath == "/stg2_spikesorting_outputs/analyzer_output"
	assert inputs.preproc_seg_sources_reldir == "/preprocess_outputs/per_segment_recordings"

	runtime_path_fallback = tmp_path / "runtime_fallback.yml"
	runtime_path_fallback.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  waveforms:
			    ms_before: 1.1
			    ms_after: 2.2
			    max_spikes_per_unit: 777
			  templates:
			    execution:
			      force_restart: false
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs_fallback = load_templates_inputs_from_runtime(config_path=str(runtime_path_fallback))
	assert inputs_fallback.waveform_extraction.ms_before == 1.1
	assert inputs_fallback.waveform_extraction.ms_after == 2.2
	assert inputs_fallback.waveform_extraction.max_spikes_per_unit == 777


def test_load_templates_config_supports_legacy_segment_sources_alias(tmp_path: Path) -> None:
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
			      inputs:
			        preprocessed_segments_reldir: /canonical/segments
			        preproc_seg_sources_reldir: /legacy/segments
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.preprocessed_segments_reldir == "/canonical/segments"
	assert inputs.preproc_seg_sources_reldir == "/canonical/segments"

	runtime_legacy_only_path = tmp_path / "runtime_legacy_only.yml"
	runtime_legacy_only_path.write_text(
		dedent(
			f"""
			data: {data_path}
			stages:
			  templates:
			    execution:
			      inputs:
			        preproc_seg_sources_reldir: /legacy/segments
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs_legacy_only = load_templates_inputs_from_runtime(config_path=str(runtime_legacy_only_path))
	assert inputs_legacy_only.preprocessed_segments_reldir == "/legacy/segments"
	assert inputs_legacy_only.preproc_seg_sources_reldir == "/legacy/segments"


def test_load_templates_config_parses_execution_quality_checks_block(tmp_path: Path) -> None:
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
			      quality_checks:
			        enable: true
			        check_for_multiple_peaks_at_channel_templates:
			          prominence_fraction: 0.42
			          min_separation_samples: 11
			          max_peaks_per_channel: 2
			    outputs:
			      data_outputs:
			        quality_checks:
			          multiple_peaks_at_channel_templates:
			            write_json: false
			            json_relpath: qc/run_level_multiple_peaks.json
			      per_unit_outputs:
			        quality_checks:
			          multiple_peaks_at_channel_templates:
			            write_json: false
			            json_relpath: qc/unit_multiple_peaks.json
			            plot:
			              write_png: false
			              write_svg: true
			              relpath: qc/violating_channels
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	qc = inputs.quality_checks
	assert qc.enable is True
	assert qc.suppress_warnings is False
	assert qc.check_for_multiple_peaks_at_channel_templates.enable is True
	assert qc.check_for_multiple_peaks_at_channel_templates.prominence_fraction == 0.42
	assert qc.check_for_multiple_peaks_at_channel_templates.min_separation_samples == 11
	assert qc.check_for_multiple_peaks_at_channel_templates.max_peaks_per_channel == 2

	run_out = inputs.quality_checks_outputs.check_for_multiple_peaks_at_channel_templates
	assert run_out.write_json is False
	assert run_out.json_relpath == "qc/run_level_multiple_peaks.json"

	unit_out = inputs.per_unit_outputs.quality_checks.check_for_multiple_peaks_at_channel_templates
	assert unit_out.write_json is False
	assert unit_out.json_relpath == "qc/unit_multiple_peaks.json"
	assert unit_out.plot.write_png is False
	assert unit_out.plot.write_svg is True
	assert unit_out.plot.relpath == "qc/violating_channels"
	assert unit_out.plot.show_multiple_peak_markers is False
	assert unit_out.plot.delay_peak_marker_color == "black"


def test_load_templates_config_parses_quality_check_warning_suppression_and_analysis_propagation_ordering(tmp_path: Path) -> None:
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
			      quality_checks:
			        enable: true
			        surpress_warnings: true
			      analysis:
			        propagation_ordering:
			          enable: true
			          latency_mode: negative_peak
			          debug: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.quality_checks.suppress_warnings is True
	assert inputs.per_unit_outputs.propagation_plots.ordering_latency_mode == "negative_peak"
	assert inputs.per_unit_outputs.propagation_plots.debug_ordering is True


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


def test_load_templates_config_parses_analyzer_cache_subdirs_and_require_flags(tmp_path: Path) -> None:
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
			      spikeinterface:
			        template_extraction:
			          sources:
			            include_concat: true
			            include_segments: true
			            require_concat: true
			            require_segments: true
			    outputs:
			      analyzer_cache:
			        enabled: true
			        relpath_root: cache/analyzers
			        concat_analyzer_subdir: concat_custom
			        segment_analyzers_subdir: segments_custom
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_templates_inputs_from_runtime(config_path=str(runtime_path))
	assert inputs.analyzer_cache.enabled is True
	assert inputs.analyzer_cache.relpath == "cache/analyzers"
	assert inputs.analyzer_cache.relpath_root == "cache/analyzers"
	assert inputs.analyzer_cache.concat_analyzer_subdir == "concat_custom"
	assert inputs.analyzer_cache.segment_analyzers_subdir == "segments_custom"
	assert inputs.require_concat_analyzer is True
	assert inputs.require_segment_analyzers is True
