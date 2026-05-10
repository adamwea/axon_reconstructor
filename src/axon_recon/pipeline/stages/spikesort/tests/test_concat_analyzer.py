from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from axon_recon.pipeline.stages.spikesort.config import (
    DEFAULT_SPIKESORT_PHASE_SEQUENCE,
    parse_spikesort_stage_config,
)
from axon_recon.pipeline.stages.spikesort.core.concat_analyzer import (
    DEFAULT_CONCAT_ANALYZER_EXTENSIONS,
    FINGERPRINT_FILENAME,
    compute_sorter_output_fingerprint,
    fingerprints_match,
    run_concat_analyzer_phase,
)
from axon_recon.runtime_config import RuntimeConfig


LOGGER = logging.getLogger("test_concat_analyzer")


class _FakeAnalyzer:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.computed_extensions: dict[str, dict[str, Any]] = {}
        self._has = set()

    def compute(self, extension_name: str, **kwargs: Any) -> None:
        self.computed_extensions[extension_name] = dict(kwargs)
        self._has.add(extension_name)
        # Drop a marker file inside the analyzer folder so on-disk presence
        # mirrors what spikeinterface would do.
        ext_dir = self.folder / "extensions" / extension_name
        ext_dir.mkdir(parents=True, exist_ok=True)
        (ext_dir / "params.json").write_text(json.dumps(kwargs, sort_keys=True), encoding="utf-8")

    def has_extension(self, extension_name: str) -> bool:
        return extension_name in self._has


def _fake_create_sorting_analyzer(*, sorting: Any, recording: Any, format: str, folder: Path, **kwargs: Any) -> _FakeAnalyzer:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sorter.json").write_text(json.dumps({"sorting_id": id(sorting)}, sort_keys=True), encoding="utf-8")
    (folder / "recording.json").write_text(json.dumps({"recording_id": id(recording), "format": format}, sort_keys=True), encoding="utf-8")
    analyzer = _FakeAnalyzer(folder)
    return analyzer


def _fake_load_sorting_analyzer(folder: Path) -> _FakeAnalyzer:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    analyzer = _FakeAnalyzer(folder)
    extensions_root = folder / "extensions"
    if extensions_root.exists():
        for child in extensions_root.iterdir():
            if child.is_dir():
                analyzer._has.add(child.name)
    return analyzer


def _seed_sorter_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
    (root / "spike_times.npy").write_bytes(b"\x00" * 64)
    (root / "spike_clusters.npy").write_bytes(b"\x01" * 32)


def test_compute_sorter_output_fingerprint_is_deterministic(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    _seed_sorter_output(sorter)
    fp_a = compute_sorter_output_fingerprint(sorter)
    fp_b = compute_sorter_output_fingerprint(sorter)
    assert fp_a["combined_sha256"] == fp_b["combined_sha256"]
    assert fp_a["file_count"] == 3
    assert fp_a["total_bytes"] == fp_b["total_bytes"]


def test_compute_sorter_output_fingerprint_changes_on_mutation(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    _seed_sorter_output(sorter)
    fp_before = compute_sorter_output_fingerprint(sorter)
    (sorter / "params.py").write_text("sample_rate = 99999\n", encoding="utf-8")
    fp_after = compute_sorter_output_fingerprint(sorter)
    assert fp_before["combined_sha256"] != fp_after["combined_sha256"]
    assert not fingerprints_match(fp_before, fp_after)


def test_concat_analyzer_builds_once_and_skips_unchanged(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    analyzer_dir = tmp_path / "concat_analyzer"
    _seed_sorter_output(sorter)

    first = run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        extensions={"random_spikes": {"max_spikes_per_unit": 250}, "waveforms": {"ms_before": 1.0, "ms_after": 2.0}},
        rebuild_on_sorter_output_change=True,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
        logger=LOGGER,
    )
    assert first["rebuilt"] is True
    assert first["rebuild_reason"] == "analyzer_dir_missing"
    assert sorted(first["extensions_computed"]) == ["random_spikes", "waveforms"]
    assert (analyzer_dir / FINGERPRINT_FILENAME).exists()

    second = run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        extensions={"random_spikes": {"max_spikes_per_unit": 250}, "waveforms": {"ms_before": 1.0, "ms_after": 2.0}},
        rebuild_on_sorter_output_change=True,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
        logger=LOGGER,
    )
    assert second["rebuilt"] is False
    assert second["rebuild_reason"] is None


def test_concat_analyzer_rebuilds_when_sorter_output_changes(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    analyzer_dir = tmp_path / "concat_analyzer"
    _seed_sorter_output(sorter)

    run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        rebuild_on_sorter_output_change=True,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )

    # Mutate sorter_output → next call must rebuild.
    (sorter / "params.py").write_text("sample_rate = 99999\n", encoding="utf-8")

    second = run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        rebuild_on_sorter_output_change=True,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )
    assert second["rebuilt"] is True
    assert second["rebuild_reason"] == "sorter_output_fingerprint_changed"


def test_concat_analyzer_default_extensions_when_omitted(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    analyzer_dir = tmp_path / "concat_analyzer"
    _seed_sorter_output(sorter)

    result = run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        extensions=None,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )
    assert sorted(result["extensions_computed"]) == sorted(DEFAULT_CONCAT_ANALYZER_EXTENSIONS.keys())


def test_concat_analyzer_extensions_present_on_disk(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    analyzer_dir = tmp_path / "concat_analyzer"
    _seed_sorter_output(sorter)

    extensions = {"random_spikes": {"max_spikes_per_unit": 100}, "templates": {}}
    run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        extensions=extensions,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )
    assert (analyzer_dir / "extensions" / "random_spikes" / "params.json").exists()
    assert (analyzer_dir / "extensions" / "templates" / "params.json").exists()


def test_concat_analyzer_rebuild_when_disabled_check(tmp_path: Path) -> None:
    sorter = tmp_path / "sorter_output"
    analyzer_dir = tmp_path / "concat_analyzer"
    _seed_sorter_output(sorter)

    run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        rebuild_on_sorter_output_change=True,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )
    second = run_concat_analyzer_phase(
        sorter_output_dir=sorter,
        recording=object(),
        sorting=object(),
        analyzer_dir=analyzer_dir,
        rebuild_on_sorter_output_change=False,
        create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
        load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
    )
    assert second["rebuilt"] is True
    assert second["rebuild_reason"] == "rebuild_on_sorter_output_change_disabled"


def test_concat_analyzer_rejects_missing_sorter_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_concat_analyzer_phase(
            sorter_output_dir=tmp_path / "missing",
            recording=object(),
            sorting=object(),
            analyzer_dir=tmp_path / "analyzer",
            create_sorting_analyzer_fn=_fake_create_sorting_analyzer,
            load_sorting_analyzer_fn=_fake_load_sorting_analyzer,
        )


def test_parse_spikesort_stage_config_concat_analyzer_phase() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "concat_analyzer": {
                            "enabled": True,
                            "relpath": "alt/analyzer",
                            "format": "binary_folder",
                            "rebuild_on_sorter_output_change": False,
                            "compute_sparsity": False,
                            "n_jobs": 4,
                            "extensions": {
                                "random_spikes": {"max_spikes_per_unit": 250},
                                "waveforms": None,
                            },
                        }
                    }
                }
            }
        }
    )
    parsed = parse_spikesort_stage_config(runtime_config=cfg)
    assert parsed.concat_analyzer_enabled is True
    assert parsed.concat_analyzer_relpath == "alt/analyzer"
    assert parsed.concat_analyzer_format == "binary_folder"
    assert parsed.concat_analyzer_rebuild_on_sorter_output_change is False
    assert parsed.concat_analyzer_compute_sparsity is False
    assert parsed.concat_analyzer_n_jobs == 4
    assert parsed.concat_analyzer_extensions == {
        "random_spikes": {"max_spikes_per_unit": 250},
        "waveforms": {},
    }


def test_parse_spikesort_stage_config_concat_analyzer_defaults() -> None:
    parsed = parse_spikesort_stage_config(runtime_config=RuntimeConfig({"stages": {"spikesort": {}}}))
    assert parsed.concat_analyzer_enabled is False
    assert parsed.concat_analyzer_relpath == "concat_analyzer"
    assert parsed.concat_analyzer_format == "binary_folder"
    assert parsed.concat_analyzer_rebuild_on_sorter_output_change is True
    assert parsed.concat_analyzer_compute_sparsity is True
    assert parsed.concat_analyzer_n_jobs is None
    assert parsed.concat_analyzer_extensions is None
    assert parsed.concat_analyzer_resource_class is None


def test_parse_spikesort_stage_config_rejects_non_mapping_extension() -> None:
    cfg = RuntimeConfig(
        {
            "stages": {
                "spikesort": {
                    "phases": {
                        "concat_analyzer": {
                            "enabled": True,
                            "extensions": {"random_spikes": "not-a-mapping"},
                        }
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError):
        parse_spikesort_stage_config(runtime_config=cfg)


def test_default_phase_sequence_orders_concat_analyzer_correctly() -> None:
    seq = DEFAULT_SPIKESORT_PHASE_SEQUENCE
    assert "concat_analyzer" in seq
    snapshot_idx = seq.index("snapshot_sorter_output")
    analyzer_idx = seq.index("concat_analyzer")
    bombcell_idx = seq.index("bombcell_label")
    assert snapshot_idx < analyzer_idx < bombcell_idx
