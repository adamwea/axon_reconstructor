from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from typing import Any


_NOISY_DEBUG_LOGGER_LEVELS: dict[str, int] = {
	"matplotlib": logging.WARNING,
	"matplotlib.font_manager": logging.WARNING,
	"numba": logging.WARNING,
	"numba.core": logging.WARNING,
	"numba.core.ssa": logging.WARNING,
}

_CONSOLE_SUPPRESSED_LOGGER_NAMES: tuple[str, ...] = (
	"kilosort",
	"matplotlib",
	"numba",
)


def _logger_name_matches(*, logger_name: str, patterns: tuple[str, ...] | list[str] | set[str]) -> bool:
	for pattern in patterns:
		token = str(pattern).strip()
		if not token:
			continue
		if logger_name == token or logger_name.startswith(f"{token}."):
			return True
	return False


@contextlib.contextmanager
def _temporarily_raise_logger_levels(levels_by_logger: dict[str, int]) -> Iterator[None]:
	original_levels: list[tuple[logging.Logger, int]] = []
	for logger_name, target_level in levels_by_logger.items():
		logger = logging.getLogger(str(logger_name))
		original_levels.append((logger, int(logger.level)))
		logger.setLevel(int(target_level))
	try:
		yield
	finally:
		for logger, original_level in reversed(original_levels):
			logger.setLevel(int(original_level))


def _set_stream_handler_stream(handler: logging.StreamHandler, stream: Any) -> Any:
	try:
		return handler.setStream(stream)
	except Exception:
		original_stream = getattr(handler, "stream", None)
		handler.stream = stream
		return original_stream


@contextlib.contextmanager
def _redirect_console_stream_handlers(*, logger_names: tuple[str, ...], stream: Any) -> Iterator[None]:
	redirected_handlers: list[tuple[logging.StreamHandler, Any]] = []
	seen_handler_ids: set[int] = set()
	original_add_handler = logging.Logger.addHandler

	def _patched_add_handler(self: logging.Logger, handler: logging.Handler) -> None:
		if _logger_name_matches(logger_name=str(self.name or ""), patterns=logger_names):
			if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
				handler_id = id(handler)
				if handler_id not in seen_handler_ids:
					seen_handler_ids.add(handler_id)
					redirected_handlers.append((handler, _set_stream_handler_stream(handler, stream)))
		original_add_handler(self, handler)

	logging.Logger.addHandler = _patched_add_handler  # type: ignore[assignment]
	logger_dict = logging.Logger.manager.loggerDict
	for logger_name, candidate in list(logger_dict.items()):
		if not isinstance(candidate, logging.Logger):
			continue
		if not _logger_name_matches(logger_name=str(logger_name), patterns=logger_names):
			continue
		for handler in list(candidate.handlers):
			if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
				handler_id = id(handler)
				if handler_id in seen_handler_ids:
					continue
				seen_handler_ids.add(handler_id)
				redirected_handlers.append((handler, _set_stream_handler_stream(handler, stream)))
	for logger_name in logger_names:
		candidate = logging.getLogger(str(logger_name))
		for handler in list(candidate.handlers):
			if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
				handler_id = id(handler)
				if handler_id in seen_handler_ids:
					continue
				seen_handler_ids.add(handler_id)
				redirected_handlers.append((handler, _set_stream_handler_stream(handler, stream)))
	try:
		yield
	finally:
		logging.Logger.addHandler = original_add_handler  # type: ignore[assignment]
		for handler, original_stream in reversed(redirected_handlers):
			_set_stream_handler_stream(handler, original_stream)


@contextlib.contextmanager
def suppress_spikesort_external_debug_output(*, enabled: bool) -> Iterator[None]:
	if bool(enabled):
		yield
		return
	with contextlib.ExitStack() as stack:
		suppressed_stream = io.StringIO()
		stack.enter_context(contextlib.redirect_stdout(suppressed_stream))
		stack.enter_context(contextlib.redirect_stderr(suppressed_stream))
		stack.enter_context(_temporarily_raise_logger_levels(_NOISY_DEBUG_LOGGER_LEVELS))
		stack.enter_context(
			_redirect_console_stream_handlers(
				logger_names=_CONSOLE_SUPPRESSED_LOGGER_NAMES,
				stream=suppressed_stream,
			)
		)
		yield