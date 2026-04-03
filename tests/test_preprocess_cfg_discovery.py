from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.stg1_preprocessing.planning import (
    format_cfg_discovery_summary,
    infer_cfg_well_id,
    summarize_cfg_files_by_well,
)


def test_infer_cfg_well_id_supports_well_token_and_numeric_prefix() -> None:
    assert infer_cfg_well_id(Path("/tmp/well7_layout.cfg")) == "well007"
    assert infer_cfg_well_id(Path("/tmp/3_014.cfg")) == "well003"
    assert infer_cfg_well_id(Path("/tmp/foo_stream12_bar.cfg")) == "well012"
    assert infer_cfg_well_id(Path("/tmp/no_match.cfg")) is None


def test_summarize_cfg_files_by_well_tracks_unmatched_and_current_stream() -> None:
    cfg_files = [
        Path("/tmp/0_001.cfg"),
        Path("/tmp/1_001.cfg"),
        Path("/tmp/1_002.cfg"),
        Path("/tmp/well005_extra.cfg"),
        Path("/tmp/unmatched_name.cfg"),
    ]

    summary = summarize_cfg_files_by_well(cfg_files=cfg_files, stream_id="well001")

    assert summary["total_cfg_files"] == 5
    assert summary["matched_cfg_files"] == 4
    assert summary["unmatched_cfg_files"] == 1
    assert summary["current_stream_id"] == "well001"
    assert summary["current_stream_cfg_count"] == 2

    per_well = summary["per_well_cfg_counts"]
    assert isinstance(per_well, dict)
    assert per_well.get("well000") == 1
    assert per_well.get("well001") == 2
    assert per_well.get("well005") == 1


def test_format_cfg_discovery_summary_includes_well_counts_and_unmatched() -> None:
    summary = {
        "per_well_cfg_counts": {"well000": 19, "well001": 19},
        "unmatched_cfg_files": 2,
        "current_stream_id": "well001",
        "current_stream_cfg_count": 19,
    }

    line = format_cfg_discovery_summary(summary)

    assert "well000=19" in line
    assert "well001=19" in line
    assert "unmatched=2" in line
    assert "current_stream=well001:19" in line
