from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from axon_recon.pipeline.shared.sampling import read_maxwell_sampling_frequency_hz


@pytest.mark.parametrize(
	"stream_id, expected_hz",
	[
		("well001", 10_000.0),
		("well002", 20_000.0),
		("1", 10_000.0),
	],
)
def test_read_maxwell_sampling_frequency_hz_from_data_store(
	tmp_path: Path,
	stream_id: str,
	expected_hz: float,
) -> None:
	h5py = pytest.importorskip("h5py")

	h5_path = tmp_path / "sampling_data_store.h5"
	with h5py.File(str(h5_path), "w") as h5:
		ds = h5.create_group("data_store")

		data0 = ds.create_group("data0000")
		data0.create_dataset("well_id", data=np.asarray([1], dtype=np.int32))
		settings0 = data0.create_group("settings")
		settings0.create_dataset("sampling", data=np.asarray([10_000.0], dtype=np.float64))

		data1 = ds.create_group("data0001")
		data1.create_dataset("well_id", data=np.asarray([2], dtype=np.int32))
		settings1 = data1.create_group("settings")
		settings1.create_dataset("sampling", data=np.asarray([20_000.0], dtype=np.float64))

	out = read_maxwell_sampling_frequency_hz(h5_path=h5_path, stream_id=stream_id)
	assert out is not None
	assert float(out) == pytest.approx(expected_hz)


def test_read_maxwell_sampling_frequency_hz_falls_back_to_attrs(tmp_path: Path) -> None:
	h5py = pytest.importorskip("h5py")

	h5_path = tmp_path / "sampling_attrs.h5"
	with h5py.File(str(h5_path), "w") as h5:
		wells = h5.create_group("wells")
		stream = wells.create_group("well001")
		stream.attrs["sampling_frequency"] = 25_000.0

	out = read_maxwell_sampling_frequency_hz(h5_path=h5_path, stream_id="well001")
	assert out is not None
	assert float(out) == pytest.approx(25_000.0)


def test_read_maxwell_sampling_frequency_hz_returns_none_when_missing(tmp_path: Path) -> None:
	h5py = pytest.importorskip("h5py")

	h5_path = tmp_path / "sampling_missing.h5"
	with h5py.File(str(h5_path), "w") as h5:
		h5.create_group("wells")

	out = read_maxwell_sampling_frequency_hz(h5_path=h5_path, stream_id="well001")
	assert out is None
