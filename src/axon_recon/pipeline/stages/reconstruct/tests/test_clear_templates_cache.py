from __future__ import annotations

from pathlib import Path

from axon_recon.pipeline.stages.reconstruct.core.clear_templates_cache import run_clear_templates_cache_phase


def _write_dummy(path: Path, payload: bytes = b"dummy") -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(payload)


def _make_templates_cache(well_out_dir: Path) -> dict[str, Path]:
	cache_dir = well_out_dir / "template_outputs" / "cache"
	paths = {
		"concat": cache_dir / "analyzers" / "concat" / "dummy.bin",
		"segments": cache_dir / "analyzers" / "segments" / "dummy.bin",
		"merged_template": cache_dir / "templates" / "merged" / "unit_1" / "merged_template.npy",
		"full_template": cache_dir / "templates" / "full" / "unit_1" / "full_template.npy",
		"source_payload": cache_dir / "source_payloads" / "dummy.bin",
	}
	for key, path in paths.items():
		_write_dummy(path, payload=f"{key}\n".encode("utf-8"))
	return paths


def test_run_clear_templates_cache_phase_preserves_merged_when_requested(tmp_path: Path) -> None:
	well_out_dir = tmp_path / "well000"
	paths = _make_templates_cache(well_out_dir)
	cache_dir = well_out_dir / "template_outputs" / "cache"

	summary = run_clear_templates_cache_phase(
		well_out_dir=well_out_dir,
		enabled=True,
		keep_merged_per_unit_outputs=True,
		keep_full_channels_templates=False,
	)

	assert summary["phase"] == "clear_templates_cache"
	assert summary["skipped"] is False
	assert summary["files_deleted"] == 4
	assert summary["bytes_deleted"] > 0
	assert str((cache_dir / "templates" / "merged").resolve()) in summary["paths_preserved"]
	assert not (cache_dir / "analyzers").exists()
	assert not (cache_dir / "source_payloads").exists()
	assert not (cache_dir / "templates" / "full").exists()
	assert paths["merged_template"].exists()


def test_run_clear_templates_cache_phase_disabled_deletes_nothing(tmp_path: Path) -> None:
	well_out_dir = tmp_path / "well000"
	paths = _make_templates_cache(well_out_dir)

	summary = run_clear_templates_cache_phase(
		well_out_dir=well_out_dir,
		enabled=False,
		keep_merged_per_unit_outputs=False,
		keep_full_channels_templates=False,
	)

	assert summary == {"phase": "clear_templates_cache", "skipped": True, "reason": "disabled"}
	for path in paths.values():
		assert path.exists()
