from __future__ import annotations

from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor


def test_pipeline_driver_init_smoke() -> None:
    recon = AxonReconstructor(h5_parent_dirs=[])

    assert recon.h5_parent_dirs == []
    assert recon.mea_environment == "nersc"
    assert recon.mea_auto_run_driver is False
    assert recon.enable_checkpointing is True
    assert recon.force_restart is False


def test_iter_raw_h5_files_empty() -> None:
    recon = AxonReconstructor(h5_parent_dirs=[])
    assert recon.iter_raw_h5_files() == []
