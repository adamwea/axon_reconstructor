from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from axon_recon.pipeline.stages.preprocess.core.prepare_raw_binaries import run_prepare_raw_binaries_core


class _FakeSegmentRecording:
	def __init__(self, *, rec_name: str, frames: int, num_channels: int = 4, sampling_frequency_hz: float = 10_000.0):
		self.rec_name = str(rec_name)
		self.frames = int(frames)
		self.num_channels = int(num_channels)
		self.sampling_frequency_hz = float(sampling_frequency_hz)

	def get_sampling_frequency(self) -> float:
		return float(self.sampling_frequency_hz)

	def get_num_channels(self) -> int:
		return int(self.num_channels)

	def get_num_segments(self) -> int:
		return 1

	def get_num_frames(self, segment_index: int | None = None, **kwargs) -> int:
		_ = segment_index, kwargs
		return int(self.frames)

	def save(self, *, folder: Path, format: str, overwrite: bool, n_jobs: int, chunk_duration: str, progress_bar: bool) -> None:
		folder = Path(folder)
		folder.mkdir(parents=True, exist_ok=True)
		(folder / "recording.marker").write_text(
			json.dumps(
				{
					"format": str(format),
					"overwrite": bool(overwrite),
					"n_jobs": int(n_jobs),
					"chunk_duration": str(chunk_duration),
					"progress_bar": bool(progress_bar),
					"segment_frames": [int(self.frames)],
				}
			),
			encoding="utf-8",
		)


def test_run_prepare_raw_binaries_core_saves_one_binary_per_rec_name(tmp_path: Path, monkeypatch) -> None:
	read_calls: list[tuple[str, str, str]] = []

	def _fake_read_maxwell(*args, **kwargs):
		file_path = kwargs.get("file_path", args[0] if args else None)
		stream_id = kwargs.get("stream_id")
		rec_name = kwargs.get("rec_name")
		read_calls.append((str(file_path), str(stream_id), str(rec_name)))
		frames_by_name = {"rec0000": 100, "rec0001": 150, "rec0002": 200}
		return _FakeSegmentRecording(rec_name=str(rec_name), frames=int(frames_by_name[str(rec_name)]))

	fake_extractors = types.SimpleNamespace(read_maxwell=_fake_read_maxwell)
	monkeypatch.setitem(sys.modules, "spikeinterface.extractors", fake_extractors)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.preprocess.core.prepare_raw_binaries._ensure_maxwell_hdf5_plugin_path",
		lambda **kwargs: None,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.preprocess.core.prepare_raw_binaries._list_maxwell_recording_names",
		lambda **kwargs: (["rec0000", "rec0001", "rec0002"], None),
	)

	recording_dir = tmp_path / "raw_binary_recording"
	manifest_path = tmp_path / "context" / "raw_binary_manifest.json"
	payload = run_prepare_raw_binaries_core(
		h5_path=tmp_path / "input.raw.h5",
		source_h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		recording_dir=recording_dir,
		manifest_path=manifest_path,
		overwrite_saved_recording=True,
		n_jobs=3,
		chunk_duration="1s",
		progress_bar=True,
		logger=None,
	)

	assert read_calls == [
		(str(tmp_path / "input.raw.h5"), "well000", "rec0000"),
		(str(tmp_path / "input.raw.h5"), "well000", "rec0001"),
		(str(tmp_path / "input.raw.h5"), "well000", "rec0002"),
	]
	assert payload["segment_count"] == 3
	assert payload["rec_names"] == ["rec0000", "rec0001", "rec0002"]
	assert payload["num_frames_by_segment"] == [100, 150, 200]
	assert manifest_path.exists()
	manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
	assert manifest_payload["rec_names"] == ["rec0000", "rec0001", "rec0002"]
	assert manifest_payload["segment_count"] == 3
	assert manifest_payload["num_frames_by_segment"] == [100, 150, 200]
	segment_entries = list(manifest_payload["segments"])
	assert [str(item["rec_name"]) for item in segment_entries] == ["rec0000", "rec0001", "rec0002"]
	for expected_frames, segment_entry in zip((100, 150, 200), segment_entries, strict=True):
		segment_dir = Path(str(segment_entry["folder"]))
		marker_payload = json.loads((segment_dir / "recording.marker").read_text(encoding="utf-8"))
		assert marker_payload["segment_frames"] == [expected_frames]


def test_run_prepare_raw_binaries_core_suppresses_read_maxwell_output_when_requested(
	tmp_path: Path,
	monkeypatch,
	capsys,
) -> None:
	def _fake_read_maxwell(*args, **kwargs):
		_ = args, kwargs
		print("The h5 compression library for Maxwell is already located somewhere!")
		return _FakeSegmentRecording(rec_name="rec0000", frames=100)

	fake_extractors = types.SimpleNamespace(read_maxwell=_fake_read_maxwell)
	monkeypatch.setitem(sys.modules, "spikeinterface.extractors", fake_extractors)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.preprocess.core.prepare_raw_binaries._ensure_maxwell_hdf5_plugin_path",
		lambda **kwargs: None,
	)
	monkeypatch.setattr(
		"axon_recon.pipeline.stages.preprocess.core.prepare_raw_binaries._list_maxwell_recording_names",
		lambda **kwargs: (["rec0000"], None),
	)

	run_prepare_raw_binaries_core(
		h5_path=tmp_path / "input.raw.h5",
		source_h5_path=tmp_path / "input.raw.h5",
		stream_id="well000",
		recording_dir=tmp_path / "raw_binary_recording",
		manifest_path=tmp_path / "context" / "raw_binary_manifest.json",
		overwrite_saved_recording=True,
		n_jobs=1,
		chunk_duration="1s",
		progress_bar=False,
		suppress_h5_plugin_messages=True,
		logger=None,
	)

	captured = capsys.readouterr()
	assert "The h5 compression library for Maxwell" not in captured.out
	assert "The h5 compression library for Maxwell" not in captured.err