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
