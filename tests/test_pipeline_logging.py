from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.pipeline_logging import setup_pipeline_logger


def test_pipeline_logger_appends_and_avoids_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "000050_well000_pipeline.log"

    logger1 = setup_pipeline_logger(log_file=log_file, logger_name="axon_test.well000", verbose=True)
    logger1.info("first")

    # Re-create/ask again: should not add a second FileHandler for same file.
    logger2 = setup_pipeline_logger(log_file=log_file, logger_name="axon_test.well000", verbose=True)
    logger2.info("second")

    text = log_file.read_text(encoding="utf-8")
    assert "first" in text
    assert "second" in text

    # Separator header should only be written once.
    assert text.count("=" * 80) == 1
