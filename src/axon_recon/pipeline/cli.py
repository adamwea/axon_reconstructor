from __future__ import annotations

import argparse

from .stages.reconstruct.cli import register_reconstruct_subparser
from .stages.templates.cli import register_templates_subparser


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="axon_recon")
	subparsers = parser.add_subparsers(dest="command", required=True)

	stages_parser = subparsers.add_parser("stages", help="Run pipeline stages")
	stages_subparsers = stages_parser.add_subparsers(dest="stage", required=True)
	register_reconstruct_subparser(stages_subparsers)
	register_templates_subparser(stages_subparsers)

	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	handler = getattr(args, "handler", None)
	if handler is None:
		parser.print_help()
		return 2
	return int(handler(args))


if __name__ == "__main__":
	raise SystemExit(main())

