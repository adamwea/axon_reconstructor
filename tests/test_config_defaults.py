def test_param_merging_te_params(tmp_path, monkeypatch):
    """The reconstructor should merge default params with user overrides.

    This stays dependency-light: it doesn't require spikeinterface/h5py.
    """

    from axon_reconstructor.pipeline import reconstructor as recon_mod

    # If spikeinterface is available, keep defaults deterministic.
    if recon_mod.ss is not None:
        monkeypatch.setattr(recon_mod.ss.Kilosort2Sorter, "default_params", lambda: {})

    recon = recon_mod.AxonReconstructor(
        h5_parent_dirs=[],
        n_jobs=5,
        te_params={"upsample": 4},
        log_file=str(tmp_path / "axon_reconstruction.log"),
        error_log_file=str(tmp_path / "axon_reconstruction_error.log"),
        logger_level="CRITICAL",
    )

    assert recon.te_params["align_cutout"] is True
    assert recon.te_params["n_jobs"] == 5
    assert recon.te_params["upsample"] == 4


def test_param_merging_sorting_params(tmp_path, monkeypatch):
    from axon_reconstructor.pipeline import reconstructor as recon_mod

    if recon_mod.ss is not None:
        monkeypatch.setattr(recon_mod.ss.Kilosort2Sorter, "default_params", lambda: {})

    recon = recon_mod.AxonReconstructor(
        h5_parent_dirs=[],
        n_jobs=2,
        sorting_params={"detect_threshold": 9, "use_gpu": False},
        log_file=str(tmp_path / "axon_reconstruction.log"),
        error_log_file=str(tmp_path / "axon_reconstruction_error.log"),
        logger_level="CRITICAL",
    )

    assert recon.sorting_params["n_jobs"] == 2
    assert recon.sorting_params["detect_threshold"] == 9
    assert recon.sorting_params["use_gpu"] is False


def test_param_merging_reconstructor_options(tmp_path, monkeypatch):
    from axon_reconstructor.pipeline import reconstructor as recon_mod

    if recon_mod.ss is not None:
        monkeypatch.setattr(recon_mod.ss.Kilosort2Sorter, "default_params", lambda: {})

    recon = recon_mod.AxonReconstructor(
        h5_parent_dirs=[],
        reconstructor_save_options={"templates": False, "waveforms": True},
        reconstructor_load_options={"load_templates": False, "load_wfs": False},
        log_file=str(tmp_path / "axon_reconstruction.log"),
        error_log_file=str(tmp_path / "axon_reconstruction_error.log"),
        logger_level="CRITICAL",
    )

    assert recon.reconstructor_save_options["templates"] is False
    assert recon.reconstructor_save_options["waveforms"] is True
    assert recon.reconstructor_load_options["load_templates"] is False
    assert recon.reconstructor_load_options["load_wfs"] is False
