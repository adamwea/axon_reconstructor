from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_plugin_installer():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "containers" / "axon-recon" / "install_maxwell_hdf5_plugin.py"
    spec = importlib.util.spec_from_file_location("axon_recon_container_plugin_installer", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maxwell_plugin_installer_rejects_non_elf_download(monkeypatch, tmp_path: Path, capsys) -> None:
    installer = _load_plugin_installer()

    def _fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(b"not a shared object")

    monkeypatch.setattr(installer, "_download", _fake_download)

    with pytest.raises(SystemExit) as exc:
        installer.install_plugin(url="https://example.test/libcompression.so", plugin_dir=tmp_path)

    assert exc.value.code == 70
    assert "container needs an update" in capsys.readouterr().err
    assert not (tmp_path / "libcompression.so").exists()


def test_maxwell_plugin_installer_writes_elf_plugin(monkeypatch, tmp_path: Path) -> None:
    installer = _load_plugin_installer()

    def _fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(b"\x7fELF" + b"fake-shared-object")

    monkeypatch.setattr(installer, "_download", _fake_download)

    target = installer.install_plugin(url="https://example.test/libcompression.so", plugin_dir=tmp_path)

    assert target == tmp_path / "libcompression.so"
    assert target.read_bytes().startswith(b"\x7fELF")