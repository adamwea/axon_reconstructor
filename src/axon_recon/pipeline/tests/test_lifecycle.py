from __future__ import annotations

import signal

import pytest

from axon_recon.pipeline.execution import lifecycle


def test_termination_signal_handler_exits_with_signal_status() -> None:
    with pytest.raises(SystemExit) as exc:
        lifecycle._termination_signal_handler(signal.SIGTERM, None)

    assert exc.value.code == 128 + int(signal.SIGTERM)


def test_install_process_lifecycle_registers_termination_handlers(monkeypatch) -> None:
    registered: list[tuple[int, object]] = []

    monkeypatch.setattr(lifecycle, "_PROCESS_LIFECYCLE_INSTALLED", False)
    monkeypatch.setattr(lifecycle, "install_linux_parent_death_signal", lambda signum=signal.SIGTERM: True)
    monkeypatch.setattr(lifecycle.signal, "signal", lambda sig, handler: registered.append((sig, handler)))

    lifecycle.install_process_lifecycle()

    seen = {sig for sig, _handler in registered}
    assert signal.SIGTERM in seen
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in seen


def test_install_linux_parent_death_signal_calls_prctl(monkeypatch) -> None:
    class DummyPrctl:
        def __init__(self):
            self.argtypes = None
            self.restype = None
            self.calls: list[tuple[int, int, int, int, int]] = []

        def __call__(self, option, arg2, arg3, arg4, arg5):
            self.calls.append((option, arg2, arg3, arg4, arg5))
            return 0

    class DummyLib:
        def __init__(self):
            self.prctl = DummyPrctl()

    dummy_lib = DummyLib()
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    monkeypatch.setattr(lifecycle.ctypes, "CDLL", lambda *args, **kwargs: dummy_lib)
    monkeypatch.setattr(lifecycle.os, "getppid", lambda: 1234)

    assert lifecycle.install_linux_parent_death_signal() is True
    assert dummy_lib.prctl.calls == [(1, int(signal.SIGTERM), 0, 0, 0)]


def test_install_linux_parent_death_signal_allows_container_pid_one_parent(monkeypatch) -> None:
    class DummyPrctl:
        def __init__(self):
            self.argtypes = None
            self.restype = None
            self.calls: list[tuple[int, int, int, int, int]] = []

        def __call__(self, option, arg2, arg3, arg4, arg5):
            self.calls.append((option, arg2, arg3, arg4, arg5))
            return 0

    class DummyLib:
        def __init__(self):
            self.prctl = DummyPrctl()

    dummy_lib = DummyLib()
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    monkeypatch.setattr(lifecycle.ctypes, "CDLL", lambda *args, **kwargs: dummy_lib)
    monkeypatch.setattr(lifecycle.os, "getppid", lambda: 1)
    monkeypatch.setattr(lifecycle.os.path, "exists", lambda path: path == "/.dockerenv")
    monkeypatch.setattr(lifecycle.os, "kill", lambda *args, **kwargs: pytest.fail("worker should not self-kill"))

    assert lifecycle.install_linux_parent_death_signal() is True
    assert dummy_lib.prctl.calls == [(1, int(signal.SIGTERM), 0, 0, 0)]