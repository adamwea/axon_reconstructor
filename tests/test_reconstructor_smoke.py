from __future__ import annotations

from pathlib import Path

from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor


def test_pipeline_driver_init_smoke() -> None:
    recon = AxonReconstructor(h5_parent_dirs=[])
    assert recon.h5_parent_dirs == []


def test_iter_raw_h5_files_single_file(tmp_path: Path) -> None:
    f = tmp_path / "data.raw.h5"
    f.write_bytes(b"fake")

    recon = AxonReconstructor(h5_parent_dirs=[f])
    files = recon.iter_raw_h5_files()
    assert files == [f]
