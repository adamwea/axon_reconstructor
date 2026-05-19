"""Centralized phase checkpoint helpers.

Per `guardrails/stage_phase_architecture.md` §"Checkpoint status enum +
auto-restart-from-first-broken", every phase's ``summary_json`` carries a
``status`` field whose value is one of seven canonical tokens:

    {missing, in_progress, ok, error, stale, skipped, dry_run_ok}

This module implements the **write side** of that contract:

- ``write_in_progress_marker(...)`` lays down a stub summary_json with
  ``status: in_progress`` BEFORE the phase's main work begins. If the
  process dies mid-phase (OOM, kernel kill, ctrl-C), the marker stays as
  ``in_progress`` and the next stage invocation's auto-restart logic
  (slice 14) sees it and force-restarts that phase.

- ``write_error_summary(...)`` overwrites the marker with ``status: error``
  + traceback when the phase raises an uncaught exception.

- ``with_checkpoint_marker(...)`` is the contextmanager that wires the two
  together: writes ``in_progress`` on entry, writes ``error`` on uncaught
  exception (and re-raises). Phases that succeed overwrite the marker
  themselves with their final ``status: ok`` summary (existing behavior
  unchanged).

The read side:

- ``read_checkpoint_status(...)`` parses an existing summary_json's
  ``status`` field. Missing file → "missing". Unparsable / unrecognized →
  "error" (a successful phase always writes a recognized status).

- ``is_stale(...)`` checks whether any upstream input path has an mtime
  newer than the summary_json. Used by slice 14's status reader.

The ``marker: true`` flag in the in-progress payload is what distinguishes
a stub from a finalized summary. Final ok/error summaries (written by
each phase's existing code path) DO NOT carry ``marker: true`` — they're
full summaries with all the phase's outputs.

Backward compat: legacy summary_json files (written before this module
landed) don't have ``marker`` at all. ``read_checkpoint_status`` just reads
the ``status`` field directly and ignores the absence of ``marker`` — so
existing on-disk analyzed_data outputs remain valid without rewrite.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import traceback
from pathlib import Path
from typing import Any, Iterator, Literal


# The seven canonical statuses. Stored verbatim in the summary_json's
# ``status`` field by each phase's writer. ``stale`` is a read-side
# determination — never written to disk.
CheckpointStatus = Literal[
	"missing",
	"in_progress",
	"ok",
	"error",
	"stale",
	"skipped",
	"dry_run_ok",
]


_VALID_STATUSES: frozenset[str] = frozenset(
	{"missing", "in_progress", "ok", "error", "stale", "skipped", "dry_run_ok"}
)

# Legacy ok-equivalent verbs some phases historically emit. Treated as
# ``ok`` for the purposes of ``read_checkpoint_status`` so the auto-restart
# logic doesn't trip on benign synonyms.
_OK_ALIASES: frozenset[str] = frozenset({"ok", "success", "completed"})


def _utc_now_iso() -> str:
	return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
	"""Write payload as JSON to ``path`` via a tmp-rename for atomicity.

	The tmp file lives in the same directory so the rename is on the same
	filesystem. If anything goes wrong mid-write, the destination stays at
	its previous state — readers never see a half-written file.
	"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	# Use a pid-suffixed tmp name so concurrent writers (e.g. MPI ranks
	# racing to write the same marker) don't trip over each other's tmp
	# files. The final rename is atomic per POSIX.
	tmp_name = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
	try:
		tmp_name.write_text(
			json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
		)
		os.replace(tmp_name, path)
	finally:
		try:
			tmp_name.unlink()
		except FileNotFoundError:
			pass


def write_in_progress_marker(
	summary_json_path: Path,
	*,
	phase_name: str,
	stage_name: str | None = None,
	extra_fields: dict[str, Any] | None = None,
) -> None:
	"""Write a stub summary_json with ``status: in_progress``.

	Overwrites any existing file at the path. Phase runners call this
	BEFORE their main work begins. On clean completion the runner
	overwrites with the real ok/error summary; on hard crash the marker
	stays as ``in_progress`` so auto-restart can catch it.

	The ``marker: true`` flag distinguishes this stub from a finalized
	summary. Slice 14's auto-restart reader uses ``status`` as the primary
	signal but can fall back to ``marker`` when triaging anomalies.
	"""
	payload: dict[str, Any] = {
		"status": "in_progress",
		"phase": str(phase_name),
		"started_at": _utc_now_iso(),
		"pid": int(os.getpid()),
		"marker": True,
	}
	if stage_name is not None:
		payload["stage"] = str(stage_name)
	if extra_fields:
		payload.update(extra_fields)
	_atomic_write_json(Path(summary_json_path), payload)


def write_error_summary(
	summary_json_path: Path,
	*,
	phase_name: str,
	exc: BaseException,
	stage_name: str | None = None,
	extra_fields: dict[str, Any] | None = None,
) -> None:
	"""Write a summary_json stub with ``status: error`` + exception info.

	Called by ``with_checkpoint_marker`` on uncaught exceptions, so phases
	whose own code didn't write a final summary still surface a usable
	error record at the marker path. Carries ``marker: true`` to signal
	this is a fallback record (the phase's own success path would have
	written a full summary without the marker flag).
	"""
	payload: dict[str, Any] = {
		"status": "error",
		"phase": str(phase_name),
		"started_at": _utc_now_iso(),
		"pid": int(os.getpid()),
		"marker": True,
		"error": {
			"type": type(exc).__name__,
			"message": str(exc),
			"traceback": "".join(
				traceback.format_exception(type(exc), exc, exc.__traceback__)
			),
		},
	}
	if stage_name is not None:
		payload["stage"] = str(stage_name)
	if extra_fields:
		payload.update(extra_fields)
	_atomic_write_json(Path(summary_json_path), payload)


@contextlib.contextmanager
def with_checkpoint_marker(
	summary_json_path: Path,
	*,
	phase_name: str,
	stage_name: str | None = None,
	extra_fields: dict[str, Any] | None = None,
) -> Iterator[None]:
	"""Context manager that wraps a phase's main body with marker semantics.

	On entry: writes the in-progress marker (overwriting any prior summary).
	On exit without exception: leaves the marker in place — the phase's
	own success-path write is expected to have overwritten it with the
	final ``status: ok`` payload.
	On uncaught exception: overwrites the marker with ``status: error`` +
	traceback, then re-raises.

	The "leaves the marker on success" behavior is intentional: the
	context manager is a SUPPORT layer, not a replacement for the phase's
	existing summary writer. Each phase still writes its own canonical
	summary_json on the success path. The marker just ensures something
	always sits on disk at the summary_json location, so a hard crash
	leaves an identifiable ``in_progress`` for the next invocation.
	"""
	write_in_progress_marker(
		Path(summary_json_path),
		phase_name=phase_name,
		stage_name=stage_name,
		extra_fields=extra_fields,
	)
	try:
		yield
	except BaseException as exc:
		# Catch BaseException so KeyboardInterrupt / SystemExit also produce
		# an error marker. Re-raise without altering the exception type so
		# the calling stack sees the original.
		try:
			write_error_summary(
				Path(summary_json_path),
				phase_name=phase_name,
				stage_name=stage_name,
				exc=exc,
				extra_fields=extra_fields,
			)
		except Exception:
			# Marker write itself failed (disk full, permission denied,
			# etc.). Don't shadow the original exception; the in_progress
			# marker on disk already signals the phase didn't complete.
			pass
		raise


def read_checkpoint_status(summary_json_path: Path) -> CheckpointStatus:
	"""Return the status from an existing summary_json file.

	- Missing file → ``"missing"``.
	- Unparsable / IOError / wrong shape → ``"error"`` (a successful phase
	  always writes well-formed JSON, so corruption signals failure).
	- ``status`` field absent or unrecognized → ``"error"``.
	- Legacy ``status: "success"`` or ``"completed"`` aliases map to ``"ok"``.

	Does NOT compute staleness — use ``is_stale`` for that determination.
	A phase whose summary_json is on disk but whose inputs are newer is
	``ok`` per this reader; the caller layers staleness on top.
	"""
	path = Path(summary_json_path)
	if not path.is_file():
		return "missing"
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return "error"
	if not isinstance(payload, dict):
		return "error"
	raw = payload.get("status")
	if raw is None:
		return "error"
	token = str(raw).strip().lower()
	if token in _OK_ALIASES:
		return "ok"
	if token in _VALID_STATUSES:
		# typing: token is one of the literal values; cast for the type checker.
		return token  # type: ignore[return-value]
	return "error"


def is_stale(
	summary_json_path: Path,
	*input_paths: Path,
	extra_mtime: float | None = None,
) -> bool:
	"""Return True iff the summary_json's inputs are newer than the summary.

	A summary is stale when:
	  - the summary_json file exists, AND
	  - at least one ``input_paths`` entry exists with mtime > summary mtime,
	    OR ``extra_mtime`` (if supplied) > summary mtime.

	Returns False when:
	  - the summary_json doesn't exist (use ``read_checkpoint_status`` → "missing"),
	  - all inputs are older than (or equal to) the summary's mtime,
	  - all supplied input_paths are missing (nothing to compare against).

	``extra_mtime`` lets callers pass an external timestamp (e.g. the
	YAML's mtime) without having to materialize a file path.
	"""
	summary_path = Path(summary_json_path)
	if not summary_path.is_file():
		# Missing isn't stale; it's its own status. Caller decides what to do.
		return False
	try:
		summary_mtime = summary_path.stat().st_mtime
	except OSError:
		return False
	for raw in input_paths:
		input_path = Path(raw)
		try:
			input_mtime = input_path.stat().st_mtime
		except OSError:
			# Missing input doesn't make the summary stale; the phase will
			# fail on read anyway. Caller is responsible for "missing
			# upstream" detection.
			continue
		if input_mtime > summary_mtime:
			return True
	if extra_mtime is not None and float(extra_mtime) > summary_mtime:
		return True
	return False


def find_first_broken_phase(
	phase_sequence: tuple[str, ...] | list[str],
	summary_json_paths: dict[str, Path],
	*,
	yaml_skipped_phases: frozenset[str] | set[str] | None = None,
) -> tuple[int, CheckpointStatus] | None:
	"""Locate the first phase in a sequence that needs to run.

	Walks ``phase_sequence`` in order. For each phase:
	  - Reads its checkpoint status via ``read_checkpoint_status``.
	  - If the status is ``ok``, the phase is healthy → keep walking.
	  - If the phase appears in ``yaml_skipped_phases`` AND its status is
	    ``missing`` or ``skipped``, treat the skip as legitimate (the YAML
	    intentionally disables this phase) → keep walking.
	  - Otherwise the phase needs to run. Return its (index, status).

	Returns ``None`` when every phase is healthy (or legitimately skipped),
	meaning the stage is a no-op for the target. Slice 14 callers treat a
	non-None return as "force-restart this phase + everything downstream",
	leaving the earlier phases alone.

	``summary_json_paths`` maps phase_name → path; phases absent from the
	dict are treated as ``missing`` (always trigger restart). The
	``yaml_skipped_phases`` argument lets callers express "these phases
	have ``enabled: false`` in the YAML; don't restart on their behalf".
	"""
	yaml_skips = frozenset(yaml_skipped_phases or ())
	for index, phase_name in enumerate(phase_sequence):
		summary_path = summary_json_paths.get(str(phase_name))
		if summary_path is None:
			status: CheckpointStatus = "missing"
		else:
			status = read_checkpoint_status(summary_path)
		if status == "ok":
			continue
		if phase_name in yaml_skips and status in ("missing", "skipped"):
			# YAML disabled this phase; the absence of a summary is
			# intentional, not a sign of broken state.
			continue
		return (index, status)
	return None


__all__ = [
	"CheckpointStatus",
	"find_first_broken_phase",
	"is_stale",
	"read_checkpoint_status",
	"with_checkpoint_marker",
	"write_error_summary",
	"write_in_progress_marker",
]
