from __future__ import annotations

from pathlib import Path

import numpy as np

from axon_recon.pipeline.stages.templates.io import (
	resolve_materialized_templates_dirs,
	write_materialized_unit_templates,
)


def test_write_materialized_unit_templates_writes_expected_files(tmp_path: Path) -> None:
	templates_out_dir = tmp_path / "template_outputs"
	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(
		templates_out_dir=templates_out_dir
	)

	merged_template = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
	merged_locs = np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	full_template = np.asarray([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=float)
	full_locs = np.asarray([[0.0, 0.0], [8.75, 0.0], [17.5, 0.0]], dtype=float)

	write_materialized_unit_templates(
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_id=94,
		merged_template=merged_template,
		merged_locations_xy=merged_locs,
		full_template=full_template,
		full_locations_xy=full_locs,
	)

	merged_dir = merged_units_dir / "unit_94"
	full_dir = full_channels_templates_dir / "unit_94"
	assert (merged_dir / "merged_contributing_template.npy").exists()
	assert (merged_dir / "merged_contributing_channel_locations.npy").exists()
	assert (full_dir / "full_template.npy").exists()
	assert (full_dir / "full_channel_locations_xy.npy").exists()

	np.testing.assert_allclose(
		np.load(merged_dir / "merged_contributing_template.npy"),
		merged_template,
	)
	np.testing.assert_allclose(
		np.load(merged_dir / "merged_contributing_channel_locations.npy"),
		merged_locs,
	)
	np.testing.assert_allclose(np.load(full_dir / "full_template.npy"), full_template)
	np.testing.assert_allclose(np.load(full_dir / "full_channel_locations_xy.npy"), full_locs)
