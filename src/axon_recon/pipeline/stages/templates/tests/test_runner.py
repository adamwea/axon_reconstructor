from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.templates.models.inputs import (
	PerUnitTemplatesOutputsConfig,
	TemplatePlotConfig,
	TemplatesInputs,
)
from axon_recon.pipeline.stages.templates.runner import run_templates_stage


def _make_stage4_artifacts(well_out_dir: Path) -> None:
	merged_unit_dir = well_out_dir / "stg4_templates_outputs" / "templates" / "merged" / "unit_94"
	full_unit_dir = well_out_dir / "stg4_templates_outputs" / "templates" / "full" / "unit_94"
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
	_make_stage4_artifacts(well_out_dir)

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
	assert unit_png.exists()
	assert str(unit_png) == result.units[0].outputs.get("template_png")
