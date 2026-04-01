from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from .stages.analysis.cli import register_analysis_subparser
from .stages.reconstruct.cli import register_reconstruct_subparser
from .stages.templates.cli import register_templates_subparser


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="axon_recon")
	subparsers = parser.add_subparsers(dest="command", required=True)

	stages_parser = subparsers.add_parser("stages", help="Run pipeline stages")
	stages_subparsers = stages_parser.add_subparsers(dest="stage", required=True)
	register_analysis_subparser(stages_subparsers)
	register_reconstruct_subparser(stages_subparsers)
	register_templates_subparser(stages_subparsers)

	return parser


def _configure_runtime_logging_from_args(args: argparse.Namespace) -> None:
	default_level = logging.INFO
	default_format = "[%(levelname)s] %(message)s"
	level = default_level
	fmt = default_format

	config_path = getattr(args, "config", None)
	if config_path is not None:
		try:
			runtime_cfg = RuntimeConfig.load(Path(str(config_path)).expanduser().resolve())
			logger_block = runtime_cfg.get("global_logger", {})
			logger_block = logger_block if isinstance(logger_block, dict) else {}

			debug_block = logger_block.get("debug_mode", {})
			debug_block = debug_block if isinstance(debug_block, dict) else {}
			debug_enabled = bool(debug_block.get("enable", False))

			level_name: Any
			format_value: Any
			if debug_enabled:
				level_name = debug_block.get("level", logger_block.get("level", "DEBUG"))
				format_value = debug_block.get("format", logger_block.get("format", default_format))
			else:
				level_name = logger_block.get("level", "INFO")
				format_value = logger_block.get("format", default_format)

			level = getattr(logging, str(level_name).upper(), default_level)
			fmt = str(format_value or default_format)
		except Exception:
			level = default_level
			fmt = default_format

	root = logging.getLogger()
	if not root.handlers:
		logging.basicConfig(level=level, format=fmt)
		return

	root.setLevel(level)
	for handler in root.handlers:
		try:
			handler.setLevel(level)
			handler.setFormatter(logging.Formatter(fmt))
		except Exception:
			continue


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	_configure_runtime_logging_from_args(args)
	handler = getattr(args, "handler", None)
	if handler is None:
		parser.print_help()
		return 2
	return int(handler(args))


if __name__ == "__main__":
	raise SystemExit(main())
