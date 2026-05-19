"""Unit tests for ``axon_recon.pipeline.checkpoint``.

Covers all public functions:
  - ``write_in_progress_marker`` (writes status/pid/started_at, overwrites)
  - ``write_error_summary`` (writes status=error + traceback)
  - ``with_checkpoint_marker`` (writes in_progress on entry, error on exc)
  - ``read_checkpoint_status`` (round-trips each of the seven statuses;
    handles missing / unparsable / unrecognized files)
  - ``is_stale`` (detects newer inputs; returns False for missing summary
    or older inputs)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from axon_recon.pipeline.checkpoint import (
	is_stale,
	read_checkpoint_status,
	with_checkpoint_marker,
	write_error_summary,
	write_in_progress_marker,
)


# ----------------------------------------------------------------------------
# write_in_progress_marker
# ----------------------------------------------------------------------------


def test_write_in_progress_marker_writes_status_and_pid(tmp_path: Path) -> None:
	summary_path = tmp_path / "context" / "phase_summary.json"
	write_in_progress_marker(
		summary_path,
		phase_name="test_phase",
		stage_name="test_stage",
	)
	assert summary_path.is_file()
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["status"] == "in_progress"
	assert payload["phase"] == "test_phase"
	assert payload["stage"] == "test_stage"
	assert payload["pid"] == os.getpid()
	assert payload["marker"] is True
	# started_at is an ISO-8601 timestamp; parse roundtrip without raising.
	assert isinstance(payload["started_at"], str) and "T" in payload["started_at"]


def test_write_in_progress_marker_overwrites_existing_summary(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	# Pre-existing "ok" summary from a previous run.
	summary_path.write_text(
		json.dumps({"status": "ok", "phase": "test_phase", "outputs": {"foo": "bar"}}),
		encoding="utf-8",
	)
	write_in_progress_marker(summary_path, phase_name="test_phase")
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	# The marker overwrites the prior summary verbatim — none of the old
	# fields (outputs, etc.) survive. This is intentional: a fresh run
	# should not be confused with a stale ok summary.
	assert payload["status"] == "in_progress"
	assert payload["marker"] is True
	assert "outputs" not in payload


def test_write_in_progress_marker_extra_fields_are_merged(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	write_in_progress_marker(
		summary_path,
		phase_name="test_phase",
		extra_fields={"well_id": "well000", "dataset_index": 0},
	)
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["well_id"] == "well000"
	assert payload["dataset_index"] == 0


def test_write_in_progress_marker_creates_parent_directories(tmp_path: Path) -> None:
	deep = tmp_path / "a" / "b" / "c" / "d" / "phase_summary.json"
	assert not deep.parent.exists()
	write_in_progress_marker(deep, phase_name="test_phase")
	assert deep.is_file()


# ----------------------------------------------------------------------------
# write_error_summary
# ----------------------------------------------------------------------------


def test_write_error_summary_writes_traceback(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	try:
		raise RuntimeError("simulated failure")
	except RuntimeError as exc:
		write_error_summary(summary_path, phase_name="test_phase", exc=exc)
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["status"] == "error"
	assert payload["phase"] == "test_phase"
	assert payload["marker"] is True
	assert payload["error"]["type"] == "RuntimeError"
	assert payload["error"]["message"] == "simulated failure"
	assert "simulated failure" in payload["error"]["traceback"]


# ----------------------------------------------------------------------------
# with_checkpoint_marker
# ----------------------------------------------------------------------------


def test_with_checkpoint_marker_writes_in_progress_before_body(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	observed_status_during_body: list[str] = []
	with with_checkpoint_marker(summary_path, phase_name="test_phase", stage_name="test_stage"):
		observed_status_during_body.append(
			read_checkpoint_status(summary_path)
		)
		# Phase's own success-path write would happen here in real code.
		summary_path.write_text(
			json.dumps({"status": "ok", "phase": "test_phase", "outputs": {}}),
			encoding="utf-8",
		)
	assert observed_status_during_body == ["in_progress"]
	# On clean exit the context manager doesn't touch the file: phase's
	# own write of status=ok is what survives.
	final_payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert final_payload["status"] == "ok"


def test_with_checkpoint_marker_writes_error_on_exception(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	with pytest.raises(RuntimeError, match="boom"):
		with with_checkpoint_marker(
			summary_path,
			phase_name="test_phase",
			stage_name="test_stage",
		):
			raise RuntimeError("boom")
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["status"] == "error"
	assert payload["phase"] == "test_phase"
	assert payload["stage"] == "test_stage"
	assert payload["error"]["type"] == "RuntimeError"
	assert payload["error"]["message"] == "boom"


def test_with_checkpoint_marker_leaves_in_progress_when_body_dies_hard(
	tmp_path: Path,
) -> None:
	"""Simulate a hard kill: body never returns; marker stays as in_progress.

	We can't actually kill the process from inside a test, but the contract
	is "if the body crashes without going through the except path, the
	marker stays as in_progress on disk". We verify by exiting the body
	without raising (so the with-block falls through normally) but reading
	the marker BEFORE the phase's success-write happens.
	"""
	summary_path = tmp_path / "phase_summary.json"
	with with_checkpoint_marker(summary_path, phase_name="test_phase"):
		# Right here, a hard kill (e.g. -9) would leave summary_json as
		# in_progress on disk. Read it to confirm.
		payload = json.loads(summary_path.read_text(encoding="utf-8"))
		assert payload["status"] == "in_progress"
		assert payload["marker"] is True


def test_with_checkpoint_marker_propagates_keyboard_interrupt(tmp_path: Path) -> None:
	"""KeyboardInterrupt is a BaseException; the marker should still be written."""
	summary_path = tmp_path / "phase_summary.json"
	with pytest.raises(KeyboardInterrupt):
		with with_checkpoint_marker(summary_path, phase_name="test_phase"):
			raise KeyboardInterrupt()
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	assert payload["status"] == "error"
	assert payload["error"]["type"] == "KeyboardInterrupt"


# ----------------------------------------------------------------------------
# read_checkpoint_status
# ----------------------------------------------------------------------------


def test_read_checkpoint_status_handles_missing_file(tmp_path: Path) -> None:
	missing = tmp_path / "does_not_exist.json"
	assert read_checkpoint_status(missing) == "missing"


def test_read_checkpoint_status_handles_unparsable_file(tmp_path: Path) -> None:
	bad = tmp_path / "bad.json"
	bad.write_text("not valid json {", encoding="utf-8")
	assert read_checkpoint_status(bad) == "error"


def test_read_checkpoint_status_handles_missing_status_field(tmp_path: Path) -> None:
	no_status = tmp_path / "no_status.json"
	no_status.write_text(json.dumps({"phase": "x", "outputs": {}}), encoding="utf-8")
	assert read_checkpoint_status(no_status) == "error"


def test_read_checkpoint_status_handles_non_dict_payload(tmp_path: Path) -> None:
	non_dict = tmp_path / "non_dict.json"
	non_dict.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
	assert read_checkpoint_status(non_dict) == "error"


@pytest.mark.parametrize(
	"raw_status,expected",
	[
		("missing", "missing"),
		("in_progress", "in_progress"),
		("ok", "ok"),
		("error", "error"),
		("stale", "stale"),
		("skipped", "skipped"),
		("dry_run_ok", "dry_run_ok"),
		# Legacy aliases for "ok".
		("success", "ok"),
		("completed", "ok"),
		# Case-insensitive / whitespace tolerant.
		("OK", "ok"),
		("  in_progress  ", "in_progress"),
	],
)
def test_read_checkpoint_status_reads_each_status_value(
	tmp_path: Path, raw_status: str, expected: str
) -> None:
	path = tmp_path / "phase_summary.json"
	path.write_text(json.dumps({"status": raw_status}), encoding="utf-8")
	assert read_checkpoint_status(path) == expected


def test_read_checkpoint_status_unrecognized_status_is_error(tmp_path: Path) -> None:
	path = tmp_path / "phase_summary.json"
	path.write_text(json.dumps({"status": "not_a_real_status"}), encoding="utf-8")
	assert read_checkpoint_status(path) == "error"


# ----------------------------------------------------------------------------
# is_stale
# ----------------------------------------------------------------------------


def test_is_stale_detects_newer_input(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	input_path = tmp_path / "input.txt"
	summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
	# Sleep a touch so mtimes differ on coarse-grained filesystems.
	time.sleep(0.05)
	input_path.write_text("changed", encoding="utf-8")
	assert is_stale(summary_path, input_path) is True


def test_is_stale_returns_false_when_summary_newer(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	input_path = tmp_path / "input.txt"
	input_path.write_text("changed", encoding="utf-8")
	time.sleep(0.05)
	summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
	assert is_stale(summary_path, input_path) is False


def test_is_stale_returns_false_when_summary_missing(tmp_path: Path) -> None:
	"""Missing summary is its own status; not stale."""
	missing = tmp_path / "missing.json"
	input_path = tmp_path / "input.txt"
	input_path.write_text("anything", encoding="utf-8")
	assert is_stale(missing, input_path) is False


def test_is_stale_returns_false_when_no_inputs_supplied(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
	assert is_stale(summary_path) is False


def test_is_stale_ignores_missing_input_paths(tmp_path: Path) -> None:
	"""A missing input doesn't make the summary stale — the phase will fail
	on read of the missing input, not via the staleness path."""
	summary_path = tmp_path / "phase_summary.json"
	summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
	never_existed = tmp_path / "does_not_exist.txt"
	assert is_stale(summary_path, never_existed) is False


def test_is_stale_honors_extra_mtime(tmp_path: Path) -> None:
	summary_path = tmp_path / "phase_summary.json"
	summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
	summary_mtime = summary_path.stat().st_mtime
	# extra_mtime in the future → stale.
	assert is_stale(summary_path, extra_mtime=summary_mtime + 10.0) is True
	# extra_mtime in the past → not stale.
	assert is_stale(summary_path, extra_mtime=summary_mtime - 10.0) is False
