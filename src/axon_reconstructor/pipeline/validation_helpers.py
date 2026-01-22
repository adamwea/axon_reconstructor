from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class PipelineValidationError(RuntimeError):
	"""Raised when a required precondition for running a pipeline step is not met."""


@dataclass(frozen=True)
class ValidationIssue:
	level: str  # "error" | "warning"
	message: str


def _repo_root_from_here() -> Path:
	"""Best-effort repo root discovery from this file location."""

	here = Path(__file__).resolve()
	for candidate in [here.parent, *here.parents]:
		if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
			return candidate
	# Fallback: common layout is repo_root/src/axon_reconstructor/pipeline/validation_helpers.py
	return here.parents[3]


def validate_paths_exist(
	paths: Iterable[str | Path],
	*,
	kind: str = "path",
	must_be_file: bool = False,
	must_be_dir: bool = False,
	suffix: Optional[str] = None,
) -> None:
	"""Validate a collection of paths exist and match expected type."""

	for p in paths:
		path = Path(p).expanduser().resolve()
		if not path.exists():
			raise PipelineValidationError(f"Missing {kind}: {path}")
		if must_be_file and not path.is_file():
			raise PipelineValidationError(f"Expected {kind} to be a file, got: {path}")
		if must_be_dir and not path.is_dir():
			raise PipelineValidationError(f"Expected {kind} to be a directory, got: {path}")
		if suffix and path.suffix != suffix:
			raise PipelineValidationError(f"Expected {kind} to have suffix {suffix!r}, got: {path}")


def validate_env_vars_set(names: Iterable[str], *, allow_empty: bool = False) -> None:
	"""Validate environment variables exist (and optionally are non-empty)."""

	for name in names:
		if name not in os.environ:
			raise PipelineValidationError(f"Missing required environment variable: {name}")
		if not allow_empty and not str(os.environ.get(name, "")).strip():
			raise PipelineValidationError(f"Required environment variable is empty: {name}")


def validate_executables_available(names: Iterable[str]) -> None:
	"""Validate executables are available on PATH."""

	missing: list[str] = []
	for name in names:
		if shutil.which(name) is None:
			missing.append(name)
	if missing:
		raise PipelineValidationError(
			"Missing required executables on PATH: " + ", ".join(missing)
		)


def ensure_maxwell_hdf5_plugin_env(
	*,
	logger=None,
	strict: bool = False,
	repo_root: Optional[str | Path] = None,
) -> Optional[Path]:
	"""Ensure Maxwell HDF5 decompression plugin is configured (host-side).

	Maxwell-compressed `.raw.h5` files require the vendor HDF5 filter plugin
	(libcompression.so). In Shifter/Docker this is typically configured already;
	on the host we set `HDF5_PLUGIN_PATH`.

	Important: this must run *before* importing `h5py` (directly or indirectly).

	Returns the plugin directory if configured/found, else None.
	"""

	# Inside some containers, /entrypoint.sh typically handles plugin setup.
	if Path("/entrypoint.sh").exists():
		return None

	existing = os.environ.get("HDF5_PLUGIN_PATH")
	if existing:
		if logger is not None:
			try:
				logger.debug("HDF5_PLUGIN_PATH already set: %s", existing)
			except Exception:
				pass
		return Path(existing.split(":", 1)[0])

	root = Path(repo_root).expanduser().resolve() if repo_root else _repo_root_from_here()

	# Preferred (and now canonical) location in this repo.
	candidates = [
		root / "vendor" / "maxwell_hdf5_plugin" / "Linux",
		# Historical smoke-test location.
		root / "tools" / "smoke_tests" / "perlmutter" / "_shared" / "vendor" / "maxwell_hdf5_plugin" / "Linux",
	]

	for plugin_dir in candidates:
		plugin_so = plugin_dir / "libcompression.so"
		if plugin_dir.is_dir() and plugin_so.exists():
			os.environ["HDF5_PLUGIN_PATH"] = str(plugin_dir)
			if logger is not None:
				try:
					logger.info("Configured HDF5_PLUGIN_PATH=%s", plugin_dir)
				except Exception:
					pass
			return plugin_dir

	msg = (
		"Maxwell HDF5 plugin not configured; reading Maxwell-compressed .raw.h5 may fail. "
		"Set HDF5_PLUGIN_PATH or ensure the plugin exists at vendor/maxwell_hdf5_plugin/Linux."
	)
	if strict:
		raise PipelineValidationError(msg)
	if logger is not None:
		try:
			logger.warning(msg)
		except Exception:
			pass
	return None


def validate_python_importable(module_name: str) -> None:
	"""Validate a Python module is importable (useful for optional integrations)."""

	try:
		__import__(module_name)
	except Exception as e:
		raise PipelineValidationError(f"Python module not importable: {module_name} ({e})") from e

