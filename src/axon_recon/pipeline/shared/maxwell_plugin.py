from __future__ import annotations

import builtins
import threading
from collections.abc import Callable
from typing import Any


MAXWELL_PLUGIN_ALREADY_LOCATED_PREFIX = "The h5 compression library for Maxwell is already located in "

_PATCH_LOCK = threading.Lock()
_PRINT_LOCK = threading.RLock()


def _message_is_maxwell_plugin_already_located(print_args: tuple[Any, ...]) -> bool:
	text = " ".join(str(value) for value in print_args)
	return text.startswith(MAXWELL_PLUGIN_ALREADY_LOCATED_PREFIX)


def install_maxwell_hdf5_plugin_message_filter() -> bool:
	try:
		import neo.rawio.maxwellrawio as maxwellrawio  # type: ignore[import-not-found]
	except Exception:
		return False

	with _PATCH_LOCK:
		current = getattr(maxwellrawio, "auto_install_maxwell_hdf5_compression_plugin", None)
		if current is None or not callable(current):
			return False
		if bool(getattr(current, "_axon_recon_filters_maxwell_plugin_message", False)):
			return True

		original: Callable[..., Any] = current

		def _quiet_auto_install(*args: Any, **kwargs: Any) -> Any:
			with _PRINT_LOCK:
				original_print = builtins.print

				def _filtered_print(*print_args: Any, **print_kwargs: Any) -> Any:
					if _message_is_maxwell_plugin_already_located(tuple(print_args)):
						return None
					return original_print(*print_args, **print_kwargs)

				builtins.print = _filtered_print
				try:
					return original(*args, **kwargs)
				finally:
					builtins.print = original_print

		setattr(_quiet_auto_install, "_axon_recon_filters_maxwell_plugin_message", True)
		setattr(_quiet_auto_install, "_axon_recon_original_auto_install", original)
		maxwellrawio.auto_install_maxwell_hdf5_compression_plugin = _quiet_auto_install
		return True
