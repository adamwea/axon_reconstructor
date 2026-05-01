from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request


DEFAULT_URL = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FLinux%2Flibcompression.so&dl=1"


def _fail(message: str) -> None:
    print(
        "ERROR: Could not install the MaxWell HDF5 compression plugin. "
        f"{message} The axon-recon container needs an update.",
        file=sys.stderr,
    )
    raise SystemExit(70)


def _download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            status = int(getattr(response, "status", 200) or 200)
            if status >= 400:
                _fail(f"Plugin URL returned HTTP {status}: {url}")
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.URLError as exc:
        _fail(f"Plugin URL is unavailable: {url} ({exc})")
    except OSError as exc:
        _fail(f"Unable to write downloaded plugin to {destination}: {exc}")


def install_plugin(*, url: str, plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    target = plugin_dir / "libcompression.so"
    with tempfile.TemporaryDirectory(prefix="axon-recon-maxwell-plugin-") as tmp:
        candidate = Path(tmp) / "libcompression.so"
        _download(url, candidate)
        header = candidate.read_bytes()[:4]
        if header != b"\x7fELF":
            _fail(f"Downloaded file from {url} is not a Linux shared object")
        candidate.replace(target)
    os.chmod(target, 0o755)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the MaxWell HDF5 compression plugin")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--plugin-dir", required=True)
    args = parser.parse_args(argv)

    target = install_plugin(url=str(args.url), plugin_dir=Path(str(args.plugin_dir)))
    print(f"Installed MaxWell HDF5 compression plugin: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())