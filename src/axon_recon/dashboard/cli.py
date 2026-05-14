"""Top-level CLI for `axon-recon dashboard` + the `axon-recon-dashboard` console script.

Parses `--config`, walks the runtime target scope, loads every present
`<well>/analysis_outputs/manifest.json`, concatenates the parquet tables,
and serves the Dash app on `localhost:<port>` (default 8050).
"""

from __future__ import annotations

import argparse
import logging
import socket
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


def _is_probably_routable_lan_ip(ip: str) -> bool:
	"""Filter heuristic for the LAN URL list shown to SSH'd users.

	Drops IPs that are technically bound on the host but won't actually be
	reachable from the user's desktop:
	  - 127.0.0.0/8 (loopback)
	  - 169.254.0.0/16 (RFC 3927 link-local — autoconf only, not routed)
	  - 172.17.0.0/16 (default Docker bridge — only routable inside the host)
	If your network legitimately uses 172.17.x.x for real LAN traffic, pass
	`--host <your-ip>` explicitly to bypass the filter.
	"""
	if not ip or ":" in ip:
		return False
	if ip.startswith("127."):
		return False
	if ip.startswith("169.254."):
		return False
	if ip.startswith("172.17."):
		return False
	return True


def detect_lan_addresses() -> list[str]:
	"""Best-effort list of non-loopback IPv4 addresses this host is reachable at.

	Tries three independent methods (outgoing-interface probe, hostname
	lookup, psutil interfaces) and merges the results so SSH'd users see
	every IP their desktop browser could connect to. Returns sorted, unique.
	Filters out clearly-unroutable ranges — see `_is_probably_routable_lan_ip`.
	"""
	addresses: set[str] = set()

	# Method 1: open a UDP socket and read the local side. No packets sent.
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
			sock.connect(("10.255.255.255", 1))
			ip = sock.getsockname()[0]
			if _is_probably_routable_lan_ip(str(ip)):
				addresses.add(str(ip))
	except OSError:
		pass

	# Method 2: hostname → IPs.
	try:
		host = socket.gethostname()
		for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
			ip = info[4][0]
			if _is_probably_routable_lan_ip(str(ip)):
				addresses.add(str(ip))
	except OSError:
		pass

	# Method 3: enumerate every interface psutil can see (psutil is already a dep).
	try:
		import psutil  # type: ignore[import-not-found]

		for _iface, addrs in psutil.net_if_addrs().items():
			for addr in addrs:
				ip = getattr(addr, "address", None)
				if isinstance(ip, str) and _is_probably_routable_lan_ip(ip):
					addresses.add(ip)
	except ImportError:
		pass

	return sorted(addresses)


def _dashboard_urls(*, bind_host: str, port: int, lan_addresses: list[str]) -> list[str]:
	"""Return the set of URLs the dashboard is reachable at, given a bind host."""
	urls: list[str] = []
	if bind_host in {"0.0.0.0", "::"}:
		urls.append(f"http://127.0.0.1:{port}/")
		for ip in lan_addresses:
			urls.append(f"http://{ip}:{port}/")
	elif bind_host in {"127.0.0.1", "localhost", "::1"}:
		urls.append(f"http://{bind_host}:{port}/")
	else:
		# Explicit host name / IP — that's the URL the user is asking for.
		urls.append(f"http://{bind_host}:{port}/")
	return urls


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
	parser.add_argument(
		"--host",
		default="127.0.0.1",
		help=(
			"Bind host (default 127.0.0.1, loopback only). Use --lan or "
			"--host 0.0.0.0 to also serve over the local network."
		),
	)
	parser.add_argument(
		"--lan",
		action="store_true",
		help=(
			"Shortcut for --host 0.0.0.0; bind to all interfaces and print every "
			"LAN URL the server is reachable at (useful when SSH'd in)."
		),
	)
	parser.add_argument(
		"--no-browser",
		action="store_true",
		help=(
			"No-op kept for forward compat. The dashboard is always headless — "
			"it never opens a browser; copy/click one of the URLs printed at startup."
		),
	)
	parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode (Dash debug toolbar)")
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

	def _load_tables() -> tuple:
		"""Re-runnable closure: discover manifests + load parquets fresh.

		Used both for initial startup load and by the dashboard's "Reload
		data" button so the running server can pick up new analysis runs
		without a restart.
		"""
		manifest_paths = iter_manifest_paths_from_config(
			config_path=str(args.config),
			target_datasets=target_datasets,
			limit_wells=args.limit_wells,
			limit_datasets=args.limit_datasets,
			limit_wells_per_dataset=args.limit_wells_per_dataset,
		)
		LOGGER.info("Discovered %d manifest(s) under the requested scope", len(manifest_paths))
		tables = load_all(manifest_paths)
		_units = tables.get("units")
		_well = tables.get("well_summary")
		if _units is not None:
			LOGGER.info(
				"Loaded units=%d rows, well_summary=%d rows",
				len(_units),
				0 if _well is None else len(_well),
			)
		return _units, _well

	units_df, well_summary_df = _load_tables()

	# Local import keeps the discovery / data path importable without Dash.
	from .app import build_app

	app = build_app(units_df, well_summary_df, data_loader=_load_tables)

	bind_host = "0.0.0.0" if bool(args.lan) else str(args.host)
	lan_addresses: list[str] = []
	if bind_host == "0.0.0.0":
		lan_addresses = detect_lan_addresses()
	urls = _dashboard_urls(bind_host=bind_host, port=int(args.port), lan_addresses=lan_addresses)

	separator = "=" * 70
	LOGGER.info(separator)
	LOGGER.info("axon-recon dashboard ready (headless — no browser will be opened)")
	LOGGER.info("Bound to %s:%d. Open any of these in your browser:", bind_host, args.port)
	for url in urls:
		LOGGER.info("  %s", url)
	if bind_host == "127.0.0.1" and not bool(args.lan):
		LOGGER.info(
			"(Loopback only — for LAN/SSH access, re-run with --lan or "
			"--host 0.0.0.0.)"
		)
	LOGGER.info("Ctrl-C or SIGTERM to stop.")
	LOGGER.info(separator)

	app.run(
		host=bind_host,
		port=int(args.port),
		debug=bool(args.debug),
		use_reloader=False,
	)
	return 0


def entry_point() -> None:
	raise SystemExit(main())


if __name__ == "__main__":
	entry_point()
