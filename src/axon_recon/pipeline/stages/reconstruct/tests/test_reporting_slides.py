from __future__ import annotations

from pathlib import Path

from axon_recon.pipeline.stages.reconstruct.reporting.slides import write_reconstruct_report_markdown


def test_write_reconstruct_report_markdown_writes_expected_content(tmp_path: Path) -> None:
    out = tmp_path / "report.md"

    result = write_reconstruct_report_markdown(
        output_md=out,
        h5_path=Path("/tmp/input.raw.h5"),
        stream_id="well001",
        reconstruction_out_dir=Path("/tmp/out/recon_outputs"),
        stage_outputs={"summary_png": "/tmp/out/recon_outputs/summary.png"},
        unit_rows=[
            {
                "unit_id": 1,
                "status": "ok",
                "outputs": {"amplitude_map_png": "/tmp/out/recon_outputs/units/0001/maps/amplitude_map.png"},
                "error": None,
            },
            {
                "unit_id": 2,
                "status": "error",
                "outputs": {},
                "error": "failed",
            },
        ],
    )

    assert result == out
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "# Reconstruct Report" in text
    assert "stream_id: well001" in text
    assert "summary_png" in text
    assert "| unit_id | status | outputs | error |" in text
    assert "| 1 | ok |" in text
    assert "| 2 | error |" in text
