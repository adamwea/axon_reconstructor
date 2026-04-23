from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys
import threading


LOGGER = logging.getLogger("axon_recon.pipeline.execution.lifecycle")
_PR_SET_PDEATHSIG = 1
_INSTALL_LOCK = threading.Lock()
_PROCESS_LIFECYCLE_INSTALLED = False


def _termination_signal_handler(signum: int, _frame: object) -> None:
    raise SystemExit(128 + int(signum))


def install_linux_parent_death_signal(signum: int | signal.Signals = signal.SIGTERM) -> bool:
    if not sys.platform.startswith("linux"):
        return False

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        result = prctl(_PR_SET_PDEATHSIG, int(signum), 0, 0, 0)
        if result != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

        # Avoid the race where the parent exits before prctl is installed.
        if os.getppid() == 1:
            os.kill(os.getpid(), int(signum))
        return True
    except Exception:
        LOGGER.debug("Unable to install Linux parent-death signal", exc_info=True)
        return False


def install_process_lifecycle() -> None:
    global _PROCESS_LIFECYCLE_INSTALLED

    with _INSTALL_LOCK:
        if _PROCESS_LIFECYCLE_INSTALLED:
            return

        install_linux_parent_death_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _termination_signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _termination_signal_handler)

        _PROCESS_LIFECYCLE_INSTALLED = True