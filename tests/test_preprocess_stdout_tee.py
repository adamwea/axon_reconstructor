from __future__ import annotations

import logging
from pathlib import Path

from axon_reconstructor.pipeline.stg1_preprocessing.h5_helpers import _tee_stdout_to_file


def test_stdout_tee_survives_write_after_context_close(tmp_path: Path) -> None:
    out_path = tmp_path / "assay_stats_test.txt"
    logger = logging.getLogger("axon_reconstructor.tests.stdout_tee")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    with _tee_stdout_to_file(out_path):
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.info("inside tee")

    # Simulate a retained handler writing after the tee context has closed.
    handler.stream.write("after close still works\n")
    handler.stream.flush()

    logger.removeHandler(handler)
    handler.close()
