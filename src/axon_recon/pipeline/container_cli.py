from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_IMAGE = "axon-recon:local"
FINGERPRINT_LABEL = "org.axon-recon.source-fingerprint"
FINGERPRINT_EXCLUDED_DIRS = {
	".git",
	"__pycache__",
	".pytest_cache",
	".ruff_cache",
	".mypy_cache",
	".ipynb_checkpoints",
	"build",
	"dist",
	"outputs",
	"scratch",
}


@dataclass
class WrapperOptions:
	image: str = DEFAULT_IMAGE
	container_cli: str = "docker"
	gpu_request: str | None = None
	shm_size: str | None = "8g"
	repo_root: Path | None = None
	repo_mode: str = "ro"
	cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "axon-recon-container")
	container_user: str | None = None
	config_mounts: bool = True
	dry_run: bool = False
	tty: bool = True
	auto_build: bool = True
	build_explicit: bool = False
	force_rebuild: bool = False
	extra_mounts: list[str] = field(default_factory=list)
	extra_env: list[str] = field(default_factory=list)
	container_args: list[str] = field(default_factory=list)


def usage() -> str:
	return """Usage: axon-recon-container [wrapper-options] <axon-reconstructor-args...>

Wrapper options:
  --image IMAGE          Container image to run (default: AXON_RECON_CONTAINER_IMAGE or axon-recon:local)
  --build                Build/update the image before running, even with --image
  --no-build             Do not build/update the image before running
  --rebuild              Force a rebuild before running
  --dry-run              Print the resolved build/run commands without running them
	--gpus SPEC            Pass Docker --gpus SPEC (default: AXON_RECON_CONTAINER_GPUS when set)
	--shm-size SIZE        Pass Docker --shm-size SIZE (default: AXON_RECON_CONTAINER_SHM_SIZE or 8g)
	--no-shm-size          Do not override Docker shared memory size
	--no-gpus              Do not request container GPU access
  --repo-root PATH       Repo root to mount (default: git top-level or installed source root)
  --repo-writable        Mount the repo read-write instead of read-only
  --cache-dir PATH       Host cache directory (default: ~/.cache/axon-recon-container)
  --user SPEC            Run the container as a Docker user spec (for example UID:GID)
  --current-user         Run the container as the current host UID:GID
  --no-config-mounts     Do not inspect --config to add data/output/scratch mounts
  --mount SPEC           Extra docker -v mount, repeatable (host:container[:mode])
  --env SPEC             Extra docker -e env var, repeatable (NAME=VALUE)
  --no-tty               Do not allocate an interactive TTY
  --wrapper-help         Show this help

All remaining arguments are passed unchanged to axon-reconstructor inside the image.
The default local image is built or updated when its source fingerprint is missing or stale.
"""


def _env_bool(name: str, default: bool) -> bool:
	raw = os.environ.get(name)
	if raw is None:
		return bool(default)
	return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _run_capture(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
	try:
		return subprocess.run(
			args,
			cwd=str(cwd) if cwd is not None else None,
			check=False,
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
	except OSError as exc:
		return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))


def _logical_cwd() -> Path:
	raw = os.environ.get("PWD") or os.getcwd()
	path = Path(raw).expanduser()
	if path.exists():
		return path
	return Path.cwd()


def _detect_repo_root() -> Path:
	cwd = _logical_cwd()
	git_prefix = _run_capture(["git", "rev-parse", "--show-prefix"], cwd=cwd)
	if git_prefix.returncode == 0:
		prefix = git_prefix.stdout.strip().strip("/")
		if prefix:
			root = cwd
			for _part in Path(prefix).parts:
				root = root.parent
			return root
		return cwd

	git_root = _run_capture(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
	if git_root.returncode == 0 and git_root.stdout.strip():
		return Path(git_root.stdout.strip()).expanduser()

	for parent in Path(__file__).resolve().parents:
		if (parent / "pyproject.toml").exists() and (parent / "containers" / "axon-recon").exists():
			return parent

	return cwd.resolve()


def _parse_options(argv: list[str]) -> WrapperOptions:
	options = WrapperOptions(
		image=os.environ.get("AXON_RECON_CONTAINER_IMAGE", DEFAULT_IMAGE),
		container_cli=os.environ.get("AXON_RECON_CONTAINER_CLI", "docker"),
		gpu_request=(os.environ.get("AXON_RECON_CONTAINER_GPUS") or None),
		shm_size=(os.environ.get("AXON_RECON_CONTAINER_SHM_SIZE") or "8g"),
		cache_dir=Path(os.environ.get("AXON_RECON_CONTAINER_CACHE", Path.home() / ".cache" / "axon-recon-container")),
		container_user=os.environ.get("AXON_RECON_CONTAINER_USER") or None,
		config_mounts=_env_bool("AXON_RECON_CONTAINER_CONFIG_MOUNTS", True),
		auto_build=_env_bool("AXON_RECON_CONTAINER_AUTO_BUILD", True),
	)

	idx = 0
	while idx < len(argv):
		arg = argv[idx]
		if arg == "--image":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --image requires a value")
			options.image = argv[idx]
			idx += 1
			continue
		if arg == "--build":
			options.auto_build = True
			options.build_explicit = True
			idx += 1
			continue
		if arg == "--no-build":
			options.auto_build = False
			idx += 1
			continue
		if arg == "--rebuild":
			options.auto_build = True
			options.build_explicit = True
			options.force_rebuild = True
			idx += 1
			continue
		if arg == "--dry-run":
			options.dry_run = True
			idx += 1
			continue
		if arg == "--gpus":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --gpus requires a value")
			gpu_request = argv[idx].strip()
			if not gpu_request:
				raise SystemExit("axon-recon-container: --gpus requires a non-empty value")
			options.gpu_request = gpu_request
			idx += 1
			continue
		if arg == "--no-gpus":
			options.gpu_request = None
			idx += 1
			continue
		if arg == "--shm-size":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --shm-size requires a value")
			shm_size = argv[idx].strip()
			if not shm_size:
				raise SystemExit("axon-recon-container: --shm-size requires a non-empty value")
			options.shm_size = shm_size
			idx += 1
			continue
		if arg == "--no-shm-size":
			options.shm_size = None
			idx += 1
			continue
		if arg == "--repo-root":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --repo-root requires a value")
			repo_root = Path(argv[idx]).expanduser()
			if not repo_root.is_absolute():
				repo_root = _logical_cwd() / repo_root
			options.repo_root = Path(os.path.abspath(str(repo_root)))
			idx += 1
			continue
		if arg == "--repo-writable":
			options.repo_mode = "rw"
			idx += 1
			continue
		if arg == "--cache-dir":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --cache-dir requires a value")
			options.cache_dir = Path(argv[idx]).expanduser()
			idx += 1
			continue
		if arg == "--user":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --user requires a value")
			options.container_user = argv[idx]
			idx += 1
			continue
		if arg == "--current-user":
			options.container_user = f"{os.getuid()}:{os.getgid()}"
			idx += 1
			continue
		if arg == "--no-config-mounts":
			options.config_mounts = False
			idx += 1
			continue
		if arg == "--mount":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --mount requires a value")
			options.extra_mounts.append(argv[idx])
			idx += 1
			continue
		if arg == "--env":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --env requires a value")
			options.extra_env.append(argv[idx])
			idx += 1
			continue
		if arg == "--no-tty":
			options.tty = False
			idx += 1
			continue
		if arg == "--wrapper-help":
			print(usage(), end="")
			raise SystemExit(0)
		if arg == "--":
			idx += 1
			break
		break

	options.container_args = list(argv[idx:]) or ["--help"]
	return options


def _clean_scalar(raw: str) -> str | None:
	value = raw.split("#", 1)[0].strip()
	if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
		value = value[1:-1]
	if value in {"", "null", "None", "~"}:
		return None
	return value


def _scalar_values(path: Path, key: str) -> list[str]:
	pattern = re.compile(rf"^\s*(?:-\s*)?{re.escape(key)}\s*:\s*(.*?)\s*$")
	values: list[str] = []
	try:
		lines = path.read_text(encoding="utf-8").splitlines()
	except OSError as exc:
		raise SystemExit(f"axon-recon-container: cannot read config file {path}: {exc}") from exc
	for line in lines:
		if line.lstrip().startswith("#"):
			continue
		match = pattern.match(line)
		if match is None:
			continue
		value = _clean_scalar(match.group(1))
		if value is not None:
			values.append(value)
	return values


def _resolve_config_path(raw: str, *, base: Path) -> Path:
	path = Path(os.path.expandvars(os.path.expanduser(raw)))
	if not path.is_absolute():
		path = base / path
	return path.resolve()


def _is_under(path: Path, parent: Path) -> bool:
	try:
		path.resolve().relative_to(parent.resolve())
		return True
	except ValueError:
		return False


def _find_cli_config_path(args: list[str]) -> str | None:
	idx = 0
	while idx < len(args):
		arg = args[idx]
		if arg == "--config":
			if idx + 1 >= len(args):
				raise SystemExit("axon-recon-container: --config requires a value")
			return args[idx + 1]
		if arg.startswith("--config="):
			return arg.split("=", 1)[1]
		idx += 1
	return None


def _resolve_config_mounts(*, repo_root: Path, config_path: str) -> list[str]:
	raw_path = Path(os.path.expandvars(os.path.expanduser(config_path)))
	if not raw_path.is_absolute():
		raw_path = _logical_cwd() / raw_path
	config = raw_path.resolve()
	if not config.exists():
		raise SystemExit(f"axon-recon-container: --config path does not exist: {config}")

	mounts: list[tuple[Path, str]] = []

	def add_mount(path: Path, mode: str, label: str) -> None:
		resolved = path.resolve()
		if _is_under(resolved, repo_root):
			return
		if mode == "ro" and not resolved.exists():
			raise SystemExit(f"axon-recon-container: required {label} path does not exist: {resolved}")
		if mode == "rw":
			try:
				resolved.mkdir(parents=True, exist_ok=True)
			except OSError as exc:
				raise SystemExit(f"axon-recon-container: cannot create writable {label} path {resolved}: {exc}") from exc
		item = (resolved, mode)
		if item not in mounts:
			mounts.append(item)

	add_mount(config.parent, "ro", "runtime config directory")
	data_values = _scalar_values(config, "data")
	data_path: Path | None = None
	if data_values:
		data_path = _resolve_config_path(data_values[0], base=config.parent)
		add_mount(data_path.parent, "ro", "data config directory")

	if data_path is not None and data_path.exists():
		data_base = data_path.parent
		for key in ("output_root", "output_root_2"):
			for raw in _scalar_values(data_path, key):
				add_mount(_resolve_config_path(raw, base=data_base), "rw", key)
		for raw in _scalar_values(data_path, "scratch_root"):
			add_mount(_resolve_config_path(raw, base=data_base), "rw", "scratch_root")
		raw_parent_paths = [
			_resolve_config_path(raw, base=data_base).parent
			for raw in _scalar_values(data_path, "raw_data_h5_path")
		]
		if raw_parent_paths:
			raw_common = Path(os.path.commonpath([str(path) for path in raw_parent_paths]))
			if str(raw_common) in {"/", "/mnt", "/home"}:
				for raw_parent in raw_parent_paths:
					add_mount(raw_parent, "ro", "raw_data_h5_path parent")
			else:
				add_mount(raw_common, "ro", "raw_data_h5_path common root")

	return [f"{path}:{path}:{mode}" for path, mode in sorted(mounts, key=lambda item: str(item[0]))]


def _fingerprint_file(path: Path, *, root: Path, digest: Any) -> None:
	rel_path = path.relative_to(root).as_posix()
	digest.update(rel_path.encode("utf-8"))
	digest.update(b"\0")
	if path.is_symlink():
		digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
		digest.update(b"\0")
		return
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
		digest.update(b"\0")


def _iter_fingerprint_files(root: Path) -> list[Path]:
	files: list[Path] = []
	for dirpath, dirnames, filenames in os.walk(root):
		dirnames[:] = sorted(
			name
			for name in dirnames
			if name not in FINGERPRINT_EXCLUDED_DIRS and not name.endswith(".egg-info")
		)
		current = Path(dirpath)
		if any(part in FINGERPRINT_EXCLUDED_DIRS or part.endswith(".egg-info") for part in current.relative_to(root).parts):
			continue
		for filename in sorted(filenames):
			path = current / filename
			if path.is_file() or path.is_symlink():
				files.append(path)
	return files


def _source_fingerprint(repo_root: Path) -> str:
	digest = hashlib.sha256()
	for root in _fingerprint_roots(repo_root):
		digest.update(str(root.relative_to(repo_root.parent)).encode("utf-8", errors="surrogateescape"))
		digest.update(b"\0")
		for path in _iter_fingerprint_files(root):
			_fingerprint_file(path, root=root, digest=digest)
	return digest.hexdigest()


def _fingerprint_roots(repo_root: Path) -> list[Path]:
	roots = [repo_root]
	for sibling in (repo_root.parent / "UnitMatch" / "UnitMatchPy", repo_root.parent / "SLAy"):
		if sibling.exists() and sibling.is_dir():
			roots.append(sibling)
	return roots


def _image_label(container_cli: str, image: str, label: str) -> str | None:
	result = _run_capture(
		[container_cli, "image", "inspect", "--format", "{{ index .Config.Labels \"" + label + "\" }}", image]
	)
	if result.returncode != 0:
		return None
	value = result.stdout.strip()
	if not value or value == "<no value>":
		return None
	return value


def _build_command(*, repo_root: Path, options: WrapperOptions, fingerprint: str) -> list[str]:
	build_script = repo_root / "containers" / "axon-recon" / "build_local_image.sh"
	if not build_script.exists():
		raise SystemExit(f"axon-recon-container: missing build helper: {build_script}")
	return [
		str(build_script),
		"--image",
		options.image,
		"--docker",
		options.container_cli,
		"--extra",
		"--label",
		"--extra",
		f"{FINGERPRINT_LABEL}={fingerprint}",
	]


def _ensure_image_current(*, repo_root: Path, options: WrapperOptions) -> None:
	if not options.auto_build:
		return
	if not (options.build_explicit or options.image == DEFAULT_IMAGE):
		return
	fingerprint = _source_fingerprint(repo_root)
	current = None if options.force_rebuild else _image_label(options.container_cli, options.image, FINGERPRINT_LABEL)
	if current == fingerprint:
		if options.dry_run:
			print(f"Image is current: {options.image} ({FINGERPRINT_LABEL}={fingerprint})")
		return
	reason = "forced rebuild" if options.force_rebuild else "missing image or stale source fingerprint"
	cmd = _build_command(repo_root=repo_root, options=options, fingerprint=fingerprint)
	if options.dry_run:
		print(f"Image build/update needed: {options.image} ({reason})")
		print("Resolved build command:")
		print(shlex.join(cmd))
		return
	print(f"axon-recon-container: building {options.image} ({reason})", file=sys.stderr)
	subprocess.run(cmd, cwd=str(repo_root), check=True)


def _build_docker_run_command(*, repo_root: Path, options: WrapperOptions) -> list[str]:
	cache_dir = options.cache_dir.expanduser().resolve()
	if not options.dry_run:
		for path in (
			cache_dir,
			cache_dir / "home",
			cache_dir / "xdg",
			cache_dir / "matplotlib",
			cache_dir / "numba",
			cache_dir / "pycache",
		):
			path.mkdir(parents=True, exist_ok=True)

	auto_mounts: list[str] = []
	if options.config_mounts:
		config_path = _find_cli_config_path(options.container_args)
		if config_path:
			auto_mounts = _resolve_config_mounts(repo_root=repo_root, config_path=config_path)

	cmd = [options.container_cli, "run", "--rm"]
	if options.tty and sys.stdin.isatty():
		cmd.append("-i")
	if options.tty and sys.stdout.isatty():
		cmd.append("-t")
	if options.gpu_request:
		cmd.extend(["--gpus", str(options.gpu_request)])
	if options.shm_size:
		cmd.extend(["--shm-size", str(options.shm_size)])

	cmd.extend(
		[
			"-v",
			f"{repo_root}:{repo_root}:{options.repo_mode}",
			"-w",
			str(repo_root),
			"-v",
			f"{cache_dir}:/tmp/axon-recon-cache:rw",
			"-e",
			"AXON_RECON_CACHE_ROOT=/tmp/axon-recon-cache",
			"-e",
			"HOME=/tmp/axon-recon-cache/home",
			"-e",
			"XDG_CACHE_HOME=/tmp/axon-recon-cache/xdg",
			"-e",
			"MPLCONFIGDIR=/tmp/axon-recon-cache/matplotlib",
			"-e",
			"NUMBA_CACHE_DIR=/tmp/axon-recon-cache/numba",
			"-e",
			"PYTHONPYCACHEPREFIX=/tmp/axon-recon-cache/pycache",
		]
	)
	if options.container_user:
		cmd.extend(["--user", options.container_user])
	for mount_spec in auto_mounts:
		cmd.extend(["-v", mount_spec])
	for mount_spec in options.extra_mounts:
		cmd.extend(["-v", mount_spec])
	for env_spec in options.extra_env:
		cmd.extend(["-e", env_spec])
	cmd.extend([options.image, *options.container_args])
	return cmd


def main(argv: list[str] | None = None) -> int:
	options = _parse_options(list(sys.argv[1:] if argv is None else argv))
	repo_root = options.repo_root or _detect_repo_root()
	if not repo_root.exists():
		raise SystemExit(f"axon-recon-container: repo root does not exist: {repo_root}")
	_ensure_image_current(repo_root=repo_root, options=options)
	cmd = _build_docker_run_command(repo_root=repo_root, options=options)
	if options.dry_run:
		print("Resolved container command:")
		print(shlex.join(cmd))
		return 0
	return subprocess.call(cmd)


if __name__ == "__main__":
	raise SystemExit(main())