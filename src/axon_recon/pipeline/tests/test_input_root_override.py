"""Tests for the `--input-root` CLI flag and its process-wide override.

Per kssynth_recon_integration slice 3b data-routing decision (PATH 2,
user 2026-05-21), the loop adds an `--input-root` override that prepends
additional roots to each target's `artifact_lookup_roots`. This lets
dev_outputs/ iteration wells consume preproc + spikesort outputs that
exist only at a reference well_out_dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon_recon.pipeline.config import (
	get_input_root_override,
	set_input_root_override,
)


@pytest.fixture(autouse=True)
def _reset_input_root_override():
	set_input_root_override(None)
	yield
	set_input_root_override(None)


def test_set_input_root_override_with_single_str(tmp_path: Path) -> None:
	root = tmp_path / "reference_data"
	root.mkdir()

	set_input_root_override(str(root))

	current = get_input_root_override()
	assert current is not None
	assert len(current) == 1
	assert current[0] == root.resolve()


def test_set_input_root_override_with_list(tmp_path: Path) -> None:
	root_a = tmp_path / "ref_a"
	root_a.mkdir()
	root_b = tmp_path / "ref_b"
	root_b.mkdir()

	set_input_root_override([str(root_a), str(root_b)])

	current = get_input_root_override()
	assert current is not None
	assert len(current) == 2
	assert current[0] == root_a.resolve()
	assert current[1] == root_b.resolve()


def test_set_input_root_override_clear_with_none() -> None:
	set_input_root_override("/tmp/foo")
	assert get_input_root_override() is not None

	set_input_root_override(None)
	assert get_input_root_override() is None


def test_set_input_root_override_clear_with_empty_string() -> None:
	set_input_root_override("/tmp/foo")
	assert get_input_root_override() is not None

	set_input_root_override("")
	assert get_input_root_override() is None


def test_set_input_root_override_ignores_empty_tokens(tmp_path: Path) -> None:
	root = tmp_path / "real_ref"
	root.mkdir()

	# Filter out empty strings in the list (simulates `--input-root a,,b`
	# where the parser yielded a list that included whitespace tokens).
	set_input_root_override([str(root), "", "  "])

	current = get_input_root_override()
	assert current is not None
	assert len(current) == 1
	assert current[0] == root.resolve()


def test_input_root_override_expands_user_and_resolves(tmp_path: Path) -> None:
	root = tmp_path / "with_relative" / "child"
	root.mkdir(parents=True)
	# Use a relative-style path going up + back to verify resolve() collapses it.
	relative_like = str(root / ".." / "child")

	set_input_root_override(relative_like)

	current = get_input_root_override()
	assert current is not None
	# After resolve(), the `..` segment is collapsed.
	assert current[0] == root.resolve()


def test_input_root_override_is_a_tuple_immutable() -> None:
	set_input_root_override(["/tmp/a", "/tmp/b"])
	current = get_input_root_override()
	assert isinstance(current, tuple)


def test_input_root_override_prepends_to_artifact_lookup_roots(tmp_path: Path) -> None:
	"""Integration: confirms the process-wide override propagates through
	`select_execution_targets` into each target's `artifact_lookup_roots`,
	which is what the analyzers loader consumes via
	`_resolve_alternate_well_out_dirs`."""

	from axon_recon.pipeline.config import (
		load_pipeline_runtime_bundle,
		select_execution_targets,
	)

	ref_root = tmp_path / "reference_data"
	ref_root.mkdir()

	data_path = tmp_path / "debug.data.yml"
	data_path.write_text(
		f"""
output_root: {tmp_path}/iter_outputs
use_scratch_root: false
datasets:
  - raw_data_h5_path: /tmp/ds1.h5
    include_in_runtime: true
    wells:
      - well_id: well001
        include_in_runtime: true
""".strip()
		+ "\n",
		encoding="utf-8",
	)

	runtime_path = tmp_path / "debug.runtime.yml"
	runtime_path.write_text(
		f"""
data: {data_path}
""".strip()
		+ "\n",
		encoding="utf-8",
	)

	# Set the process-wide override BEFORE select_execution_targets fires.
	set_input_root_override(str(ref_root))

	bundle = load_pipeline_runtime_bundle(config_path=str(runtime_path))
	targets = select_execution_targets(bundle=bundle)
	assert len(targets) == 1

	target = targets[0]
	# The reference root is prepended to artifact_lookup_roots ahead of any
	# YAML-declared output_root_2 / default_lookup_roots.
	assert ref_root.resolve() in target.artifact_lookup_roots
