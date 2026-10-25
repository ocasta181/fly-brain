"""Explicit installation and verified loading of the external VF64 Metal source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

REVISION = "8dae385a16493282e2a3e90e4cd1aa1250b36dc5"
SHA256 = "2dfe74b5357459218de0898f266ff250148678506196f9ccedc43ecc31dea848"
LICENSE_SHA256 = "a2bee73b445fe21a33eb529cf11f49b28fac1fcf83cb6186ca6ebaf9735fb8a1"
SOURCE_PATH = Path("Sources/VF64Metal/Shaders/IEEE/Arithmetic.metal")
DEFAULT_ROOT = Path.home() / ".cache/eon/VF64-metal" / REVISION
_INSTALL_COMMAND = "python code/mlx_backend/vf64.py --install"
_FILES = ((SOURCE_PATH, SHA256), (Path("LICENSE"), LICENSE_SHA256))


def _root() -> Path:
    return Path(os.environ.get("VF64_SOURCE_DIR") or DEFAULT_ROOT).expanduser()


def _verify(data: bytes, expected: str, path: Path) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"VF64 integrity check failed for {path}: {actual}; expected {expected}. "
            "Existing files will not be replaced. Set VF64_SOURCE_DIR to a clean "
            f"directory and run `{_INSTALL_COMMAND}`."
        )


def _read_verified(root: Path, relative: Path, expected: str) -> bytes:
    path = root / relative
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"VF64 {REVISION} is missing {path}. Run `{_INSTALL_COMMAND}` "
            "to install the pinned source and license, or set VF64_SOURCE_DIR "
            "to a checkout containing those exact files."
        ) from error
    _verify(data, expected, path)
    return data


def shader_header() -> str:
    """Read verified source and license without installing or accessing a network."""
    root = _root()
    source = _read_verified(root, SOURCE_PATH, SHA256)
    _read_verified(root, Path("LICENSE"), LICENSE_SHA256)
    return source.decode("utf-8")


def install() -> dict[str, object]:
    """Install exact source/license bytes, refusing to overwrite changed files."""
    start = time.perf_counter()
    root = _root()
    pending = []
    for relative, expected in _FILES:
        target = root / relative
        if target.exists():
            _read_verified(root, relative, expected)
        else:
            pending.append((relative, expected))

    downloads = []
    for relative, expected in pending:
        url = f"https://raw.githubusercontent.com/Lulzx/VF64-metal/{REVISION}/{relative.as_posix()}"
        with urllib.request.urlopen(url, timeout=12) as response:
            data = response.read()
        _verify(data, expected, root / relative)
        downloads.append((relative, expected, data))

    for relative, expected, data in downloads:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                os.link(temporary, target)
            except FileExistsError:
                _read_verified(root, relative, expected)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    for relative, expected in _FILES:
        _read_verified(root, relative, expected)
    return {
        "status": "installed",
        "root": str(root),
        "revision": REVISION,
        "files": {str(relative): expected for relative, expected in _FILES},
        "setup_wall_s": time.perf_counter() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="install the pinned VF64 source")
    arguments = parser.parse_args()
    if arguments.install:
        print(json.dumps(install(), indent=2))
    else:
        parser.print_help()
