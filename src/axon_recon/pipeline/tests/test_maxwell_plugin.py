from __future__ import annotations

import sys
import types

import axon_recon.pipeline.cli as pipeline_cli
from axon_recon.pipeline.shared.maxwell_plugin import install_maxwell_hdf5_plugin_message_filter


def _install_fake_neo(monkeypatch):
	neo_mod = types.ModuleType("neo")
	rawio_mod = types.ModuleType("neo.rawio")
	maxwellrawio_mod = types.ModuleType("neo.rawio.maxwellrawio")

	def _fake_auto_install(*_args, **_kwargs):
		print("The h5 compression library for Maxwell is already located in /tmp/libcompression.so!")
		print("neo useful output")

	maxwellrawio_mod.auto_install_maxwell_hdf5_compression_plugin = _fake_auto_install
	neo_mod.rawio = rawio_mod
	rawio_mod.maxwellrawio = maxwellrawio_mod
	monkeypatch.setitem(sys.modules, "neo", neo_mod)
	monkeypatch.setitem(sys.modules, "neo.rawio", rawio_mod)
	monkeypatch.setitem(sys.modules, "neo.rawio.maxwellrawio", maxwellrawio_mod)
	return maxwellrawio_mod


def test_maxwell_plugin_message_filter_suppresses_only_known_status_line(monkeypatch, capsys) -> None:
	maxwellrawio_mod = _install_fake_neo(monkeypatch)

	assert install_maxwell_hdf5_plugin_message_filter() is True
	maxwellrawio_mod.auto_install_maxwell_hdf5_compression_plugin()

	captured = capsys.readouterr()
	assert "The h5 compression library for Maxwell" not in captured.out
	assert "neo useful output" in captured.out


def test_pipeline_cli_installs_maxwell_plugin_message_filter(monkeypatch, tmp_path) -> None:
	_install_fake_neo(monkeypatch)
	runtime_cfg = tmp_path / "runtime.yml"
	runtime_cfg.write_text("global_logger: {}\n", encoding="utf-8")
	calls: list[str] = []

	def _handler(_args):
		calls.append("spikesort")
		return 0

	monkeypatch.setitem(pipeline_cli._STAGE_HANDLERS, "spikesort", _handler)

	assert pipeline_cli.main(["stages", "spikesort", "--config", str(runtime_cfg)]) == 0
	assert calls == ["spikesort"]

	import neo.rawio.maxwellrawio as maxwellrawio

	assert bool(getattr(maxwellrawio.auto_install_maxwell_hdf5_compression_plugin, "_axon_recon_filters_maxwell_plugin_message", False))
