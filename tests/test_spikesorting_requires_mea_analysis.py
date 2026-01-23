from __future__ import annotations

from pathlib import Path

import pytest

from axon_reconstructor.pipeline.spikesorting import SpikeSortRequest, resolve_mea_sorter_output_dir


def test_resolve_mea_sorter_output_dir_requires_mea_analysis(tmp_path: Path) -> None:
    req = SpikeSortRequest(
        data_file=tmp_path / "ProjectA" / "2026-01-01" / "Chip123" / "Network" / "123456" / "data.raw.h5",
        mea_output_root=tmp_path,
        well="well000",
    )

    with pytest.raises(RuntimeError, match="requires MEA_Analysis"):
        resolve_mea_sorter_output_dir(req)
