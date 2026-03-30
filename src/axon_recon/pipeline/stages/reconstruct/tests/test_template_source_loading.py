from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from axon_recon.pipeline.stages.reconstruct.core.reconstruct import load_templates_for_unit


def test_load_templates_for_unit_supports_full_from_merged(tmp_path: Path) -> None:
	merged_units_dir = tmp_path / "merged"
	full_units_dir = tmp_path / "full"
	merged_unit_dir = merged_units_dir / "unit_94"
	full_unit_dir = full_units_dir / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)
	full_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template = np.asarray(
		[
			[-1.0, -2.0, -1.5, -0.2],
			[-0.5, -1.0, -0.7, -0.1],
			[-0.8, -1.6, -1.1, -0.1],
		],
		dtype=float,
	)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float)

	np.save(merged_unit_dir / "merged_template.npy", merged_template)
	np.save(merged_unit_dir / "merged_channel_locations.npy", merged_locs)

	full_locs = np.asarray(
		[
			[0.0, 0.0],
			[10.0, 0.0],
			[20.0, 0.0],
			[0.0, 18.0],
			[10.0, 18.0],
			[20.0, 18.0],
		],
		dtype=float,
	)
	np.save(full_unit_dir / "full_channel_locations_xy.npy", full_locs)

	plot_template, plot_locs, gtr_template, gtr_locs, fs_hz, selected_source = load_templates_for_unit(
		unit_id=94,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_units_dir,
		template_source="full_from_merged",
		use_full_channels_templates=True,
		require_full_channels_templates=True,
	)

	assert selected_source == "full_from_merged_per_unit"
	assert plot_template.shape == (4, 3)
	assert plot_locs.shape == (3, 2)
	assert gtr_template.shape == (6, 4)
	assert gtr_locs.shape == (6, 2)
	assert fs_hz == 10_000.0

	# Mapped channels should land on grid-aligned positions, with zeros in missing channels.
	np.testing.assert_allclose(gtr_template[0, :], merged_template[0, :])
	np.testing.assert_allclose(gtr_template[2, :], merged_template[1, :])
	np.testing.assert_allclose(gtr_template[4, :], merged_template[2, :])
	assert np.allclose(gtr_template[1, :], 0.0)


def test_load_templates_for_unit_square_from_merged_handles_channel_by_time_input(tmp_path: Path) -> None:
	merged_units_dir = tmp_path / "merged"
	full_units_dir = tmp_path / "full"
	merged_unit_dir = merged_units_dir / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template_ch_by_t = np.asarray(
		[
			[-1.0, -2.0, -1.5, -0.2],
			[-0.5, -1.0, -0.7, -0.1],
			[-0.8, -1.6, -1.1, -0.1],
		],
		dtype=float,
	)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float)

	np.save(merged_unit_dir / "merged_template.npy", merged_template_ch_by_t)
	np.save(merged_unit_dir / "merged_channel_locations.npy", merged_locs)

	_, _, gtr_template, gtr_locs, _, selected_source = load_templates_for_unit(
		unit_id=94,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_units_dir,
		template_source="square",
		use_full_channels_templates=False,
		require_full_channels_templates=False,
	)

	assert selected_source == "square_from_merged_per_unit"
	assert gtr_template.shape == (9, 4)
	assert gtr_locs.shape == (9, 2)
	np.testing.assert_allclose(gtr_template[0, :], merged_template_ch_by_t[0, :])
	np.testing.assert_allclose(gtr_template[2, :], merged_template_ch_by_t[1, :])
	np.testing.assert_allclose(gtr_template[4, :], merged_template_ch_by_t[2, :])
	assert np.allclose(gtr_template[1, :], 0.0)


def test_load_templates_for_unit_square_from_merged_handles_time_by_channel_input(tmp_path: Path) -> None:
	merged_units_dir = tmp_path / "merged"
	full_units_dir = tmp_path / "full"
	merged_unit_dir = merged_units_dir / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template_ch_by_t = np.asarray(
		[
			[-1.0, -2.0, -1.5, -0.2],
			[-0.5, -1.0, -0.7, -0.1],
			[-0.8, -1.6, -1.1, -0.1],
		],
		dtype=float,
	)
	merged_template_t_by_ch = np.asarray(merged_template_ch_by_t.T, dtype=float)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float)

	np.save(merged_unit_dir / "merged_template.npy", merged_template_t_by_ch)
	np.save(merged_unit_dir / "merged_channel_locations.npy", merged_locs)

	_, _, gtr_template, gtr_locs, _, selected_source = load_templates_for_unit(
		unit_id=94,
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_units_dir,
		template_source="square",
		use_full_channels_templates=False,
		require_full_channels_templates=False,
	)

	assert selected_source == "square_from_merged_per_unit"
	assert gtr_template.shape == (9, 4)
	assert gtr_locs.shape == (9, 2)
	np.testing.assert_allclose(gtr_template[0, :], merged_template_ch_by_t[0, :])
	np.testing.assert_allclose(gtr_template[2, :], merged_template_ch_by_t[1, :])
	np.testing.assert_allclose(gtr_template[4, :], merged_template_ch_by_t[2, :])
	assert np.allclose(gtr_template[1, :], 0.0)


def test_load_templates_for_unit_square_from_merged_raises_on_incompatible_template_shape(tmp_path: Path) -> None:
	merged_units_dir = tmp_path / "merged"
	full_units_dir = tmp_path / "full"
	merged_unit_dir = merged_units_dir / "unit_94"
	merged_unit_dir.mkdir(parents=True, exist_ok=True)

	merged_template_bad = np.ones((5, 6), dtype=float)
	merged_locs = np.asarray([[0.0, 0.0], [20.0, 0.0], [10.0, 18.0]], dtype=float)

	np.save(merged_unit_dir / "merged_template.npy", merged_template_bad)
	np.save(merged_unit_dir / "merged_channel_locations.npy", merged_locs)

	with pytest.raises(ValueError, match="Cannot orient merged template to channel-by-time"):
		load_templates_for_unit(
			unit_id=94,
			merged_units_dir=merged_units_dir,
			full_channels_templates_dir=full_units_dir,
			template_source="square",
			use_full_channels_templates=False,
			require_full_channels_templates=False,
		)
