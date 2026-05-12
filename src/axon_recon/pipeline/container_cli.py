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

from axon_recon.pipeline.resources import (
	ContainerCapsConfig,
	ResourceProfileConfig,
	ResourcesConfig,
	get_active_resource_profile,
	parse_resources_config,
)
from axon_recon.runtime_config import RuntimeConfig


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
	shm_size_overridden: bool = False
	memory: str | None = None
	memory_reservation: str | None = None
	memory_swap: str | None = None
	ipc: str | None = None
	cpuset_cpus: str | None = None
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
	mpi_ranks: int = 1


def usage() -> str:
	return """Usage: axon-recon-container [wrapper-options] <axon-reconstructor-args...>

Wrapper options:
  --image IMAGE          Container image to run (default: AXON_RECON_CONTAINER_IMAGE or axon-recon:local)
  --build                Build/update the image before running, even with --image
  --no-build             Do not build/update the image before running
  --rebuild              Force a rebuild before running
  --dry-run              Print the resolved build/run commands without running them
	--gpus SPEC            Pass Docker --gpus SPEC (default: AXON_RECON_CONTAINER_GPUS when set)
	--shm-size SIZE        Pass Docker --shm-size SIZE (default: resources.container_caps.shm_size, else AXON_RECON_CONTAINER_SHM_SIZE or 8g)
	--no-shm-size          Do not override Docker shared memory size
	--no-gpus              Do not request container GPU access
	--cpuset-cpus SPEC     Pass Docker --cpuset-cpus SPEC (e.g. 0-7) to restrict visible CPUs
  --repo-root PATH       Repo root to mount (default: git top-level or installed source root)
  --repo-writable        Mount the repo read-write instead of read-only
  --cache-dir PATH       Host cache directory (default: ~/.cache/axon-recon-container)
	--user SPEC            Run the container as a Docker user spec (default: current host UID:GID on POSIX)
	--current-user         Explicitly run the container as the current host UID:GID
  --no-config-mounts     Do not inspect --config to add data/output/scratch mounts
  --mount SPEC           Extra docker -v mount, repeatable (host:container[:mode])
  --env SPEC             Extra docker -e env var, repeatable (NAME=VALUE)
  --no-tty               Do not allocate an interactive TTY
  --mpi-ranks N, -n N    Run mpirun -np N inside one container (default: 1, single rank).
                         When N>1 the inner command becomes
                         `mpirun -np N --allow-run-as-root --bind-to none axon-reconstructor ...`
                         Host `mpirun -np N axon-recon-container ...` is unsupported; the
                         wrapper owns the rank count.
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


def _host_uid_gid_user_spec() -> str | None:
	getuid = getattr(os, "getuid", None)
	getgid = getattr(os, "getgid", None)
	if not callable(getuid) or not callable(getgid):
		return None
	try:
		return f"{int(getuid())}:{int(getgid())}"
	except OSError:
		return None


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
		container_user=(os.environ.get("AXON_RECON_CONTAINER_USER") or _host_uid_gid_user_spec()),
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
		if arg == "--cpuset-cpus":
			idx += 1
			if idx >= len(argv):
				raise SystemExit("axon-recon-container: --cpuset-cpus requires a value")
			cpuset_cpus = argv[idx].strip()
			if not cpuset_cpus:
				raise SystemExit("axon-recon-container: --cpuset-cpus requires a non-empty value")
			options.cpuset_cpus = cpuset_cpus
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
			options.shm_size_overridden = True
			idx += 1
			continue
		if arg == "--no-shm-size":
			options.shm_size = None
			options.shm_size_overridden = True
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
			container_user = _host_uid_gid_user_spec()
			if container_user is None:
				raise SystemExit("axon-recon-container: --current-user is not supported on this platform")
			options.container_user = container_user
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
		if arg in {"--mpi-ranks", "-n"}:
			idx += 1
			if idx >= len(argv):
				raise SystemExit(f"axon-recon-container: {arg} requires a value")
			raw = str(argv[idx]).strip()
			try:
				ranks = int(raw)
			except Exception as exc:
				raise SystemExit(
					f"axon-recon-container: {arg} requires an integer >= 1, got {raw!r}"
				) from exc
			if ranks < 1:
				raise SystemExit(
					f"axon-recon-container: {arg} requires an integer >= 1, got {ranks}"
				)
			options.mpi_ranks = ranks
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


def _resolve_cli_config_path(config_path: str) -> Path:
	raw_path = Path(os.path.expandvars(os.path.expanduser(config_path)))
	if not raw_path.is_absolute():
		raw_path = _logical_cwd() / raw_path
	return raw_path.resolve()


def _wrapper_config_scalar(raw: str) -> str | None:
	value = str(raw).split("#", 1)[0].strip()
	if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
		value = value[1:-1]
	if value.strip().lower() in {"", "null", "none", "~"}:
		return None
	return value


def _yaml_scalar_entry(text: str, *, key_path: tuple[str, ...]) -> tuple[bool, str | None]:
	stack: list[tuple[int, str]] = []
	for raw_line in str(text or "").splitlines():
		if raw_line.lstrip().startswith("#"):
			continue
		line = raw_line.split("#", 1)[0].rstrip()
		if not line.strip() or line.lstrip().startswith("-"):
			continue
		match = re.match(r"^(\s*)([^:#][^:]*?)\s*:\s*(.*?)\s*$", line)
		if match is None:
			continue
		indent = len(match.group(1))
		key = str(match.group(2)).strip()
		value = match.group(3)
		while stack and indent <= stack[-1][0]:
			stack.pop()
		current_path = tuple(item[1] for item in stack) + (key,)
		if value == "":
			stack.append((indent, key))
			continue
		if current_path == key_path:
			return True, _wrapper_config_scalar(value)
	return False, None


def _parse_wrapper_int(
	value: str | None,
	*,
	config_path: Path,
	key_path: tuple[str, ...],
) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception as exc:
		raise SystemExit(
			"axon-recon-container: invalid integer for "
			f"{'.'.join(key_path)} in {config_path}: {value!r}"
		) from exc


def _fallback_resources_config_from_yaml(config_path: Path) -> ResourcesConfig:
	try:
		text = config_path.read_text(encoding="utf-8")
	except OSError as exc:
		raise SystemExit(f"axon-recon-container: cannot read config file {config_path}: {exc}") from exc

	_active_profile_found, active_profile = _yaml_scalar_entry(
		text,
		key_path=("resources", "active_profile"),
	)
	profiles: dict[str, ResourceProfileConfig] = {}
	if active_profile is not None:
		_analyzer_slots_found, analyzer_slots_value = _yaml_scalar_entry(
			text,
			key_path=("resources", "profiles", str(active_profile), "analyzer_slots"),
		)
		analyzer_slots = _parse_wrapper_int(
			analyzer_slots_value,
			config_path=config_path,
			key_path=("resources", "profiles", str(active_profile), "analyzer_slots"),
		)
		profiles[str(active_profile)] = ResourceProfileConfig(analyzer_slots=max(0, int(analyzer_slots or 0)))

	shm_size_configured, shm_size = _yaml_scalar_entry(
		text,
		key_path=("resources", "container_caps", "shm_size"),
	)
	memory_configured, memory = _yaml_scalar_entry(
		text,
		key_path=("resources", "container_caps", "memory"),
	)
	memory_reservation_configured, memory_reservation = _yaml_scalar_entry(
		text,
		key_path=("resources", "container_caps", "memory_reservation"),
	)
	memory_swap_configured, memory_swap = _yaml_scalar_entry(
		text,
		key_path=("resources", "container_caps", "memory_swap"),
	)
	ipc_configured, ipc = _yaml_scalar_entry(
		text,
		key_path=("resources", "container_caps", "ipc"),
	)
	return ResourcesConfig(
		active_profile=(str(active_profile) if active_profile is not None else None),
		profiles=profiles,
		container_caps=ContainerCapsConfig(
			shm_size=shm_size,
			shm_size_configured=bool(shm_size_configured),
			memory=memory,
			memory_configured=bool(memory_configured),
			memory_reservation=memory_reservation,
			memory_reservation_configured=bool(memory_reservation_configured),
			memory_swap=memory_swap,
			memory_swap_configured=bool(memory_swap_configured),
			ipc=ipc,
			ipc_configured=bool(ipc_configured),
		),
	)


def _requires_pyyaml_fallback(exc: BaseException) -> bool:
	return "PyYAML" in str(exc)


def _load_resources_config_from_config(config_path: str | None) -> ResourcesConfig | None:
	if not config_path:
		return None
	resolved = _resolve_cli_config_path(config_path)
	if not resolved.exists():
		raise SystemExit(f"axon-recon-container: --config path does not exist: {resolved}")
	try:
		runtime_config = RuntimeConfig.load(resolved)
		return parse_resources_config(runtime_config=runtime_config)
	except Exception as exc:
		if resolved.suffix.lower() in {".yml", ".yaml"} and _requires_pyyaml_fallback(exc):
			return _fallback_resources_config_from_yaml(resolved)
		raise SystemExit(f"axon-recon-container: cannot load runtime config {resolved}: {exc}") from exc


def _load_container_caps_from_config(config_path: str | None) -> ContainerCapsConfig:
	resources = _load_resources_config_from_config(config_path)
	return ContainerCapsConfig() if resources is None else resources.container_caps


def _resolve_effective_container_caps(
	*,
	options: WrapperOptions,
	container_caps: ContainerCapsConfig,
) -> dict[str, str | None]:
	shm_size = options.shm_size
	if not options.shm_size_overridden and container_caps.shm_size_configured:
		shm_size = container_caps.shm_size
	return {
		"shm_size": shm_size,
		"memory": options.memory if options.memory is not None else container_caps.memory,
		"memory_reservation": (
			options.memory_reservation
			if options.memory_reservation is not None
			else container_caps.memory_reservation
		),
		"memory_swap": options.memory_swap if options.memory_swap is not None else container_caps.memory_swap,
		"ipc": options.ipc if options.ipc is not None else container_caps.ipc,
	}


def _size_spec_to_bytes(value: str | None) -> int | None:
	if value is None:
		return None
	text = str(value).strip().lower()
	if not text:
		return None
	match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgtp]?)(?:i?b?)?", text)
	if match is None:
		return None
	amount = float(match.group(1))
	unit = str(match.group(2) or "")
	multiplier = {
		"": 1,
		"k": 1024,
		"m": 1024**2,
		"g": 1024**3,
		"t": 1024**4,
		"p": 1024**5,
	}.get(unit)
	if multiplier is None:
		return None
	return int(amount * float(multiplier))


def _targets_reconstruct_analyzers(container_args: list[str]) -> bool:
	return any(str(arg).strip() == "reconstruct.analyzers" for arg in container_args)


def _parallel_analyzer_preflight_warnings(
	*,
	resources: ResourcesConfig | None,
	container_args: list[str],
	shm_size: str | None,
	ipc: str | None,
) -> list[str]:
	if resources is None or not _targets_reconstruct_analyzers(container_args):
		return []
	if str(ipc or "").strip().lower() == "host":
		return []
	profile = get_active_resource_profile(resources)
	if profile is None:
		return []
	analyzer_slots = max(0, int(profile.analyzer_slots or 0))
	if analyzer_slots <= 1:
		return []
	recommended_shm_gib = max(8, analyzer_slots * 8)
	recommended_shm_bytes = int(recommended_shm_gib) * (1024**3)
	actual_shm_bytes = _size_spec_to_bytes(shm_size)
	if actual_shm_bytes is not None and actual_shm_bytes >= recommended_shm_bytes:
		return []
	current_shm = "the container runtime default" if shm_size is None else str(shm_size)
	profile_name = str(resources.active_profile or "unknown")
	return [
		(
			"container preflight warning: active profile "
			f"{profile_name!r} targets reconstruct.analyzers with analyzer_slots={analyzer_slots}, "
			f"but /dev/shm is {current_shm}. Parallel SpikeInterface analyzer builds use shared memory and may "
			f"exhaust this cap; consider setting resources.container_caps.shm_size (or --shm-size) to at least "
			f"{recommended_shm_gib}g, using ipc=host, or lowering analyzer_slots to 1."
		)
	]


def _container_preflight_warnings(options: WrapperOptions) -> list[str]:
	config_path = _find_cli_config_path(options.container_args)
	resources = _load_resources_config_from_config(config_path)
	if resources is None:
		return []
	effective_caps = _resolve_effective_container_caps(options=options, container_caps=resources.container_caps)
	return _parallel_analyzer_preflight_warnings(
		resources=resources,
		container_args=options.container_args,
		shm_size=effective_caps.get("shm_size", None),
		ipc=effective_caps.get("ipc", None),
	)


def _resolve_config_mounts(*, repo_root: Path, config_path: str) -> list[str]:
	config = _resolve_cli_config_path(config_path)
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
	config_path = _find_cli_config_path(options.container_args)
	container_caps = _load_container_caps_from_config(config_path)
	if options.config_mounts and config_path:
		auto_mounts = _resolve_config_mounts(repo_root=repo_root, config_path=config_path)

	effective_caps = _resolve_effective_container_caps(options=options, container_caps=container_caps)
	shm_size = effective_caps.get("shm_size", None)
	memory = effective_caps.get("memory", None)
	memory_reservation = effective_caps.get("memory_reservation", None)
	memory_swap = effective_caps.get("memory_swap", None)
	ipc = effective_caps.get("ipc", None)

	cmd = [options.container_cli, "run", "--rm"]
	if options.tty and sys.stdin.isatty():
		cmd.append("-i")
	if options.tty and sys.stdout.isatty():
		cmd.append("-t")
	if options.gpu_request:
		cmd.extend(["--gpus", str(options.gpu_request)])
	if shm_size:
		cmd.extend(["--shm-size", str(shm_size)])
	if memory:
		cmd.extend(["--memory", str(memory)])
	if memory_reservation:
		cmd.extend(["--memory-reservation", str(memory_reservation)])
	if memory_swap:
		cmd.extend(["--memory-swap", str(memory_swap)])
	if ipc:
		cmd.extend(["--ipc", str(ipc)])
	if options.cpuset_cpus:
		cmd.extend(["--cpuset-cpus", str(options.cpuset_cpus)])

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
	if options.mpi_ranks > 1:
		cmd.extend(
			[
				options.image,
				"mpirun",
				"-np",
				str(options.mpi_ranks),
				"--allow-run-as-root",
				"--bind-to",
				"none",
				"axon-reconstructor",
				*options.container_args,
			]
		)
	else:
		cmd.extend([options.image, *options.container_args])
	return cmd


def main(argv: list[str] | None = None) -> int:
	options = _parse_options(list(sys.argv[1:] if argv is None else argv))
	repo_root = options.repo_root or _detect_repo_root()
	if not repo_root.exists():
		raise SystemExit(f"axon-recon-container: repo root does not exist: {repo_root}")
	_ensure_image_current(repo_root=repo_root, options=options)
	for warning in _container_preflight_warnings(options):
		print(f"axon-recon-container: {warning}", file=sys.stderr)
	cmd = _build_docker_run_command(repo_root=repo_root, options=options)
	if options.dry_run:
		print("Resolved container command:")
		print(shlex.join(cmd))
		return 0
	return subprocess.call(cmd)


if __name__ == "__main__":
	raise SystemExit(main())