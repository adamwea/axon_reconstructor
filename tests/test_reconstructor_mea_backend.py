from __future__ import annotations

from pathlib import Path

import pytest


def test_spikesort_recordings_mea_analysis_backend_loads_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure AxonReconstructor can populate sortings from an existing MEA_Analysis-style folder.

    This test avoids requiring spikeinterface; we only verify that the expected
    `sorting_path` is recorded and that we don't fall back to internal sorting.
    """

    from axon_reconstructor.pipeline.reconstructor import AxonReconstructor

    # Create a plausible MEA_Analysis output layout: <well_output_dir>/sorter_output/<some file>
    well_output_dir = tmp_path / "outputs" / "proj" / "date" / "run" / "file.h5" / "well001"
    sorter_output_dir = well_output_dir / "sorter_output"
    sorter_output_dir.mkdir(parents=True)
    (sorter_output_dir / "spike_times.npy").write_bytes(b"\x93NUMPY")

    # Patch the integration pathing so this test doesn't depend on the exact contract.
    import axon_reconstructor.integrations.mea_analysis as mea

    monkeypatch.setattr(mea, "compute_mea_output_dir", lambda **_: well_output_dir)

    recon = AxonReconstructor(
        h5_parent_dirs=[],
        mea_environment="nersc",
        mea_analysis_output_root=tmp_path,
        mea_auto_run_driver=False,
    )

    # Avoid trying to load an existing reconstructor object.
    recon.reconstructor_load_options["load_reconstructor"] = False

    rec_key = "2020_Chip_Run"
    recon.multirecordings = {
        rec_key: {
            "h5_path": "/some/path/file.h5",
            "scanType": "AxonTracking",
            "date": "2020",
            "chip_id": "Chip",
            "run_id": "Run",
            "streams": {
                "well001": {"multirecording": None, "common_el": [], "multirec_save_path": ""}
            },
        }
    }
    recon.recordings = {rec_key: {"h5_path": "/some/path/file.h5", "streams": {}}}

    recon.spikesort_recordings()

    assert rec_key in recon.sortings
    assert "well001" in recon.sortings[rec_key]["streams"]
    entry = recon.sortings[rec_key]["streams"]["well001"]
    assert Path(entry["sorting_path"]) == well_output_dir
