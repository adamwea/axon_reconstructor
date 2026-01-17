import pytest


def test_reconstructor_init_smoke(tmp_path, monkeypatch):
    """Fast sanity check: the core pipeline object can be constructed.

    This test avoids touching real data and writes logs into a temp directory.
    """

    # Import here so the test file itself is importable even if deps are missing.
    from axon_reconstructor.pipeline.reconstructor import AxonReconstructor

    recon = AxonReconstructor(
        h5_parent_dirs=[],
        log_file=str(tmp_path / "axon_reconstruction.log"),
        error_log_file=str(tmp_path / "axon_reconstruction_error.log"),
        logger_level="CRITICAL",
    )

    assert recon.h5_parent_dirs == []
    assert recon.sorting_params["detect_threshold"] == 7
    assert recon.sorting_params["n_jobs"] == recon.n_jobs


def test_setup_logger_allows_filename_only(tmp_path, monkeypatch):
    """Regression test for log paths with no directory component.

    If log_file is just a filename, setup_logger should not try to os.makedirs('')
    and should create the files in the current working directory.
    """

    from axon_reconstructor.pipeline.reconstructor import AxonReconstructor
    monkeypatch.chdir(tmp_path)

    recon = AxonReconstructor(
        h5_parent_dirs=[],
        log_file="axon_reconstruction.log",
        error_log_file="axon_reconstruction_error.log",
        logger_level="CRITICAL",
    )

    assert (tmp_path / "axon_reconstruction.log").exists()
    assert (tmp_path / "axon_reconstruction_error.log").exists()

    # Avoid pytest 'unused variable' warning in some configs
    assert recon.logger is not None
