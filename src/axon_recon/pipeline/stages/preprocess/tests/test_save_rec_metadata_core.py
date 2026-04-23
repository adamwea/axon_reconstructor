from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np


class _FakeDataset:
	def __init__(self, value):
		self._value = value

	@property
	def dtype(self):
		return getattr(self._value, "dtype", None)

	@property
	def shape(self):
		return getattr(self._value, "shape", ())

	def __getitem__(self, key):
		if key != ():
			raise KeyError(key)
		return self._value


class _FakeGroup:
	def __init__(self, mapping: dict[str, object] | None = None, *, attrs: dict[str, object] | None = None):
		self._mapping = dict(mapping or {})
		self.attrs = dict(attrs or {})

	def __contains__(self, key: str) -> bool:
		if "/" in str(key):
			try:
				self[str(key)]
			except Exception:
				return False
			return True
		return str(key) in self._mapping

	def __getitem__(self, key: str):
		if "/" in str(key):
			node = self
			for part in str(key).split("/"):
				node = node[part]
			return node
		return self._mapping[str(key)]

	def get(self, key: str, default=None):
		return self._mapping.get(str(key), default)

	def keys(self):
		return self._mapping.keys()


def _build_fake_h5_root() -> _FakeGroup:
	mapping_dtype = np.dtype([
		("channel", np.int32),
		("electrode", np.int32),
		("x", np.float64),
		("y", np.float64),
	])
	return _FakeGroup(
		{
			"wells": _FakeGroup(
				{
					"well001": _FakeGroup(
						{
							"rec0000": _FakeGroup(
								{
									"start_time": _FakeDataset(np.asarray([1_000], dtype=np.int64)),
									"stop_time": _FakeDataset(np.asarray([1_100], dtype=np.int64)),
									"settings": _FakeGroup(
										{
											"sampling": _FakeDataset(np.asarray([10_000.0], dtype=float)),
											"mapping": _FakeDataset(
												np.asarray(
													[
														(0, 1, 0.0, 0.0),
														(1, 2, 1.0, 0.0),
														(2, 3, 2.0, 0.0),
														(3, 10, 3.0, 0.0),
													],
													dtype=mapping_dtype,
												)
											),
										}
									),
									"groups": _FakeGroup(
										{
											"routed": _FakeGroup(
												{
													"channels": _FakeDataset(np.asarray([0, 1, 2, 3], dtype=np.int64)),
													"raw": _FakeDataset(np.zeros((4, 5), dtype=np.uint16)),
													"frame_nos": _FakeDataset(np.asarray([10, 11, 12, 15, 16], dtype=np.int64)),
													"triggered": _FakeDataset(np.asarray([1], dtype=np.int64)),
													"trigger_pre": _FakeDataset(np.asarray([2], dtype=np.int64)),
													"trigger_post": _FakeDataset(np.asarray([3], dtype=np.int64)),
													"trigger_minamp": _FakeDataset(np.asarray([0.5], dtype=float)),
													"trigger_maxamp": _FakeDataset(np.asarray([1.5], dtype=float)),
												}
											)
										}
									),
								}
							),
							"rec0001": _FakeGroup(
								{
									"start_time": _FakeDataset(np.asarray([1_200], dtype=np.int64)),
									"stop_time": _FakeDataset(np.asarray([1_280], dtype=np.int64)),
									"settings": _FakeGroup(
										{
											"sampling": _FakeDataset(np.asarray([10_000.0], dtype=float)),
											"mapping": _FakeDataset(
												np.asarray(
													[
														(0, 2, 0.0, 1.0),
														(1, 3, 1.0, 1.0),
														(2, 11, 2.0, 1.0),
													],
													dtype=mapping_dtype,
												)
											),
										}
									),
									"groups": _FakeGroup(
										{
											"routed": _FakeGroup(
												{
													"channels": _FakeDataset(np.asarray([0, 1, 2], dtype=np.int64)),
													"raw": _FakeDataset(np.zeros((3, 4), dtype=np.uint16)),
													"frame_nos": _FakeDataset(np.asarray([30, 31, 32, 33], dtype=np.int64)),
													"triggered": _FakeDataset(np.asarray([0], dtype=np.int64)),
												}
											)
										}
									),
								}
							),
						}
					)
				}
			),
			"assay": _FakeGroup({"inputs": _FakeGroup({}), "run_id": _FakeDataset(1), "script_id": _FakeDataset(2)}),
			"data_store": _FakeGroup(
				{
					"data0000": _FakeGroup(
						{
							"start_time": _FakeDataset(np.asarray([5_000], dtype=np.int64)),
							"stop_time": _FakeDataset(np.asarray([5_125], dtype=np.int64)),
							"well_id": _FakeDataset(np.asarray([1], dtype=np.int64)),
							"settings": _FakeGroup({"sampling": _FakeDataset(np.asarray([10_000.0], dtype=float))}),
						}
					)
				}
			),
		}
	)


def test_run_save_rec_metadata_core_reuses_single_h5_open_and_segment_si_reads(monkeypatch, tmp_path: Path):
	from axon_recon.pipeline.stages.preprocess.core import save_rec_metadata as core

	counts = {"h5_open": 0, "read_maxwell": 0}
	root = _build_fake_h5_root()

	class _FakeH5File:
		def __enter__(self):
			counts["h5_open"] += 1
			return root

		def __exit__(self, exc_type, exc, tb):
			return False

	h5py_mod = types.ModuleType("h5py")
	h5py_mod.File = lambda *args, **kwargs: _FakeH5File()

	monkeypatch.setitem(sys.modules, "h5py", h5py_mod)
	monkeypatch.setattr(
		core,
		"find_common_electrodes_from_segments",
		lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected common-electrode reload")),
	)

	payload = core.run_save_rec_metadata_core(
		h5_path=tmp_path / "data.raw.h5",
		source_h5_path=tmp_path / "data.raw.h5",
		requested_metadata_source="source_h5",
		metadata_source="source_h5",
		stream_id="well001",
		segment_epochs_path=tmp_path / "segment_epochs.json",
		contiguous_epochs_path=tmp_path / "contiguous_epochs.json",
		sampling_metadata_path=tmp_path / "sampling_metadata.json",
		assay_stats_path=tmp_path / "assay_stats.txt",
		common_electrodes_path=tmp_path / "common_electrodes.npy",
		verbose=True,
		suppress_h5_plugin_messages=True,
		logger=None,
		report_step_timers=True,
	)

	assert counts["h5_open"] == 1
	assert counts["read_maxwell"] == 0
	assert payload["segment_count"] == 2
	assert payload["common_electrode_count"] == 2
	assert payload["total_elapsed_s"] >= 0.0
	assert payload["common_electrodes_preview"] == [2, 3]
	assert any(item["step"] == "build_segment_and_contiguous_epochs" for item in payload["step_timers"])
	assert any(item["step"] == "write_contiguous_epochs_json" for item in payload["step_timers"])
	assert payload["recording_info"] == {
		"num_segments": 2,
		"sampling_frequency_hz": 10_000.0,
		"num_frames_total": 9,
		"num_frames_by_segment": [5, 4],
		"duration_s_total": 0.0009,
		"dtype": "uint16",
	}
	assert "recording_info_error" not in payload

	segment_epochs = json.loads((tmp_path / "segment_epochs.json").read_text(encoding="utf-8"))
	assert [entry["rec_name"] for entry in segment_epochs["segments"]] == ["rec0000", "rec0001"]
	assert segment_epochs["segments"][0]["frame_no_start"] == 10
	assert segment_epochs["segments"][0]["frame_no_end"] == 16

	sampling_metadata = json.loads((tmp_path / "sampling_metadata.json").read_text(encoding="utf-8"))
	assert sampling_metadata["sampling_summary"]["stream_sampling_frequency_hz"] == 10_000.0
	assert sampling_metadata["sampling_summary"]["all_segments_match"] is True

	contiguous_epochs = json.loads((tmp_path / "contiguous_epochs.json").read_text(encoding="utf-8"))
	assert contiguous_epochs["contiguous_epoch_count"] == 3

	assay_stats = (tmp_path / "assay_stats.txt").read_text(encoding="utf-8")
	assert "assay settings: /assay keys=['inputs', 'run_id', 'script_id']" in assay_stats
	assert "data_store: stream=data0000 well=well001 start_ms=5000 stop_ms=5125 dur_s=0.125" in assay_stats
	assert payload["slowest_segments"][0]["rec_name"] in {"rec0000", "rec0001"}

	common_electrodes = np.load(tmp_path / "common_electrodes.npy")
	assert common_electrodes.tolist() == [2, 3]