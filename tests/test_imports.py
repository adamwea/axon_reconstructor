def test_import() -> None:
    import axon_reconstructor

    assert hasattr(axon_reconstructor, "__version__")
