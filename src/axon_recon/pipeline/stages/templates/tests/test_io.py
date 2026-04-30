from __future__ import annotations

from pathlib import Path

import numpy as np

from axon_recon.pipeline.stages.templates.io import (
	load_materialized_source_payload,
	resolve_materialized_source_payload_unit_dir,
	resolve_materialized_templates_dirs,
	read_json,
	write_json,
	write_materialized_source_payload,
	write_materialized_unit_templates,
)


def test_write_materialized_unit_templates_writes_expected_files(tmp_path: Path) -> None:
	templates_out_dir = tmp_path / "template_outputs"
	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(
		templates_out_dir=templates_out_dir
	)
	assert merged_units_dir == templates_out_dir / "cache" / "templates" / "merged"
	assert full_channels_templates_dir == templates_out_dir / "cache" / "templates" / "full"

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


def test_write_materialized_unit_templates_removes_full_files_when_disabled(tmp_path: Path) -> None:
	templates_out_dir = tmp_path / "template_outputs"
	merged_units_dir, full_channels_templates_dir = resolve_materialized_templates_dirs(
		templates_out_dir=templates_out_dir
	)

	full_dir = full_channels_templates_dir / "unit_94"
	full_dir.mkdir(parents=True, exist_ok=True)
	np.save(full_dir / "full_template.npy", np.asarray([[9.0, 9.0]], dtype=float))
	np.save(full_dir / "full_channel_locations_xy.npy", np.asarray([[0.0, 0.0]], dtype=float))

	write_materialized_unit_templates(
		merged_units_dir=merged_units_dir,
		full_channels_templates_dir=full_channels_templates_dir,
		unit_id=94,
		merged_template=np.asarray([[1.0, 2.0]], dtype=float),
		merged_locations_xy=np.asarray([[0.0, 0.0]], dtype=float),
		full_template=np.asarray([[1.0, 2.0]], dtype=float),
		full_locations_xy=np.asarray([[0.0, 0.0]], dtype=float),
		write_full_template=False,
	)

	merged_dir = merged_units_dir / "unit_94"
	assert (merged_dir / "merged_contributing_template.npy").exists()
	assert not full_dir.exists()


def test_materialized_source_payload_round_trip(tmp_path: Path) -> None:
	templates_out_dir = tmp_path / "template_outputs"
	template = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
	locations = np.asarray([[0.0, 0.0], [17.5, 0.0]], dtype=float)
	overlay = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)

	unit_dir = write_materialized_source_payload(
		templates_out_dir=templates_out_dir,
		output_rel_root="cache/source_payloads",
		source_name="concat",
		unit_id=94,
		template_c_by_t=template,
		locations_xy=locations,
		electrode_ids=[100, 101],
		channel_ids=[0, 1],
		waveform_count=7,
		sampling_rate_hz=10_000.0,
		overlay_waveforms=overlay,
		top_electrode_id=101,
		total_waveforms_at_channel=5,
	)

	assert unit_dir == resolve_materialized_source_payload_unit_dir(
		templates_out_dir=templates_out_dir,
		output_rel_root="cache/source_payloads",
		source_name="concat",
		unit_id=94,
	)

	payload = load_materialized_source_payload(source_payload_unit_dir=unit_dir)
	assert payload is not None
	template_rt, locations_rt, electrode_ids_rt, channel_ids_rt, waveform_count_rt, sampling_rate_rt, overlay_rt, top_id_rt, top_count_rt = payload

	np.testing.assert_allclose(template_rt, template)
	np.testing.assert_allclose(locations_rt, locations)
	assert electrode_ids_rt == [100, 101]
	assert channel_ids_rt == [0, 1]
	assert waveform_count_rt == 7
	assert sampling_rate_rt == 10_000.0
	np.testing.assert_allclose(overlay_rt, overlay)
	assert top_id_rt == 101
	assert top_count_rt == 5


def test_materialized_source_payload_serializes_numpy_scalar_ids(tmp_path: Path) -> None:
	templates_out_dir = tmp_path / "template_outputs"
	unit_dir = write_materialized_source_payload(
		templates_out_dir=templates_out_dir,
		output_rel_root="cache/source_payloads",
		source_name="concat",
		unit_id=np.int64(94),
		template_c_by_t=np.asarray([[1.0, 2.0]], dtype=float),
		locations_xy=np.asarray([[0.0, 0.0]], dtype=float),
		electrode_ids=[np.int64(100)],
		channel_ids=[np.int64(0)],
		waveform_count=3,
		sampling_rate_hz=10_000.0,
		overlay_waveforms=None,
		top_electrode_id=np.int64(100),
		total_waveforms_at_channel=2,
	)

	payload = load_materialized_source_payload(source_payload_unit_dir=unit_dir)
	assert payload is not None
	_, _, electrode_ids_rt, channel_ids_rt, _, _, _, top_id_rt, _ = payload
	assert electrode_ids_rt == [100]
	assert channel_ids_rt == [0]
	assert top_id_rt == 100


def test_write_json_serializes_nested_numpy_scalars(tmp_path: Path) -> None:
	path = tmp_path / "payload.json"
	write_json(
		path,
		{
			"unit_ids": [np.int64(94)],
			"meta": {"top_id": np.int64(101)},
		},
	)
	payload = read_json(path)
	assert payload == {"unit_ids": [94], "meta": {"top_id": 101}}
