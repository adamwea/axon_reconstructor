"""Top-level CLI for `axon-recon dashboard` + the `axon-recon-dashboard` console script.

Parses `--config`, walks the runtime target scope, loads every present
`<well>/analysis_outputs/manifest.json`, concatenates the parquet tables,
and serves the Dash app on `localhost:<port>` (default 8050).
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .data import load_all
from .discovery import iter_manifest_paths_from_config


LOGGER = logging.getLogger("axon_recon.dashboard")


def _parse_positive_int(raw: str) -> int:
	try:
		value = int(str(raw).strip())
	except Exception as exc:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw!r}") from exc
	if value <= 0:
		raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
	return value


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="axon-recon-dashboard",
		description="Serve a Plotly Dash dashboard over per-well analysis_outputs/ artifacts.",
	)
	parser.add_argument("--config", required=True, help="Path to runtime YAML/JSON config")
	parser.add_argument(
		"--target-dataset",
		"--target-datasets",
		nargs="+",
		default=None,
		dest="target_datasets",
		help="0-based dataset indices to load (e.g. --target-dataset 0 2 11)",
	)
	parser.add_argument(
		"--limit-wells",
		type=_parse_positive_int,
		default=None,
		dest="limit_wells",
		help="Limit total wells loaded across all datasets",
	)
	parser.add_argument(
		"--limit-datasets",
		type=_parse_positive_int,
		default=None,
		help="Limit number of datasets loaded",
	)
	parser.add_argument(
		"--limit-wells-per-dataset",
		type=_parse_positive_int,
		default=None,
		help="Limit wells per dataset",
	)
	parser.add_argument("--port", type=_parse_positive_int, default=8050, help="Bind port (default 8050)")
	parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
	parser.add_argument("--no-browser", action="store_true", help="Don't try to open a browser (CI default)")
	parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode (auto-reload)")
	return parser


def _parse_target_dataset_indices(raw: list[str] | None) -> list[int] | None:
	if raw is None:
		return None
	parsed: list[int] = []
	seen: set[int] = set()
	for item in raw:
		for token in str(item).split(","):
			text = str(token).strip()
			if not text:
				continue
			try:
				value = int(text)
			except ValueError as exc:
				raise SystemExit(f"Invalid dataset index for --target-dataset: {text!r}") from exc
			if value < 0:
				raise SystemExit(f"Dataset indices must be >= 0, got {value}")
			if value in seen:
				continue
			seen.add(value)
			parsed.append(value)
	return parsed or None


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_arg_parser()
	args = parser.parse_args(argv)

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

	target_datasets = _parse_target_dataset_indices(args.target_datasets)
	manifest_paths = iter_manifest_paths_from_config(
		config_path=str(args.config),
		target_datasets=target_datasets,
		limit_wells=args.limit_wells,
		limit_datasets=args.limit_datasets,
		limit_wells_per_dataset=args.limit_wells_per_dataset,
	)
	LOGGER.info("Discovered %d manifest(s) under the requested scope", len(manifest_paths))
	tables = load_all(manifest_paths)
	units_df = tables.get("units")
	well_summary_df = tables.get("well_summary")
	if units_df is not None:
		LOGGER.info(
			"Loaded units=%d rows, well_summary=%d rows",
			len(units_df),
			0 if well_summary_df is None else len(well_summary_df),
		)

	# Local import keeps the discovery / data path importable without Dash.
	from .app import build_app

	app = build_app(units_df, well_summary_df)
	LOGGER.info(
		"Starting dashboard at http://%s:%d  (--no-browser=%s, --debug=%s)",
		args.host,
		args.port,
		args.no_browser,
		args.debug,
	)
	app.run(
		host=str(args.host),
		port=int(args.port),
		debug=bool(args.debug),
		use_reloader=False,
	)
	return 0


def entry_point() -> None:
	raise SystemExit(main())


if __name__ == "__main__":
	entry_point()
