"""Pipeline entrypoints for axon_recon."""


def main(argv: list[str] | None = None) -> int:
	# Lazy import avoids preloading ``axon_recon.pipeline.cli`` before
	# ``python -m axon_recon.pipeline.cli`` executes it as ``__main__``.
	from .cli import main as _main

	return int(_main(argv))


__all__ = ["main"]

