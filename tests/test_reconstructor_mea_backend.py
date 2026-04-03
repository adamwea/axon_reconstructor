from __future__ import annotations

from pathlib import Path

import pytest


def test_spikesort_recordings_mea_analysis_backend_loads_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve sorter_output dir via MEA_Analysis contract when installed.

    When MEA_Analysis isn't installed, this test is skipped.
    """

    pytest.importorskip("IPNAnalysis.path_contract")

    from axon_reconstructor.pipeline.spikesorting import SpikeSortRequest, resolve_mea_sorter_output_dir

    # Create a plausible MEA_Analysis output layout:
    #   <well_output_dir>/spikesort_outputs/sorter_output/<some file>
    well_output_dir = tmp_path / "outputs" / "proj" / "date" / "run" / "file.h5" / "well001"
    sorter_output_dir = well_output_dir / "spikesort_outputs" / "sorter_output"
    sorter_output_dir.mkdir(parents=True)
    (sorter_output_dir / "spike_times.npy").write_bytes(b"\x93NUMPY")

    # Patch the integration pathing so this test doesn't depend on the exact contract.
    import axon_reconstructor.integrations.mea_analysis as mea

    monkeypatch.setattr(mea, "compute_sorter_output_dir", lambda **_: sorter_output_dir)

    req = SpikeSortRequest(
        data_file=tmp_path / "whatever" / "data.raw.h5",
        mea_output_root=tmp_path,
        well="well001",
    )
    resolved = resolve_mea_sorter_output_dir(req)
    assert resolved == sorter_output_dir
