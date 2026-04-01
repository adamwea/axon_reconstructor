from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from axon_recon.pipeline.stages.analysis.config import load_analysis_inputs_from_runtime


def test_load_analysis_config_from_stage_block(tmp_path: Path) -> None:
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
			  analysis:
			    execution:
			      force_restart: true
			    unit_limit: 12
			    outputs:
			      output_rel_root: analysis_v2
			      metrics:
			        per_well:
			          reldir: well_metrics/
			          n_units_total:
			            write_csv: true
			            csv_relpath: n_units_total.csv
			        cross_well:
			          statistical_testing:
			            enable: true
			            paired_wells: true
			            multiple_testing_correction:
			              enable: true
			"""
		).strip()
		+ "\n",
		encoding="utf-8",
	)

	inputs = load_analysis_inputs_from_runtime(config_path=str(runtime_path), unit_id_override=94)
	assert inputs.stream_id == "well003"
	assert inputs.output_rel_root == "analysis_v2"
	assert inputs.force_restart is True
	assert inputs.unit_limit == 12
	assert inputs.unit_ids == [94]
	assert "per_well" in inputs.metrics
	assert any("paired_wells" in msg for msg in inputs.deferred_warnings)
