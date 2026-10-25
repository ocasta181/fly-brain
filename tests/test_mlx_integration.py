"""Optional-backend routing and explicit external dependency contracts."""

import builtins
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest

from mlx_backend import vf64


@pytest.mark.parametrize("mlx,expected", [
    (False, ["cpu", "gpu", "pytorch", "nestgpu", "genn"]),
    (True, ["mlx"]),
])
def test_cli_selects_backends_without_loading_optional_runtime(monkeypatch, mlx, expected):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "run_mlx" or name.startswith(("mlx.", "mlx_backend")):
            raise AssertionError(f"CLI parsing imported optional runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "brian2cuda", ModuleType("brian2cuda"))
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "main.py"))
    main = namespace["main"]
    observed = []
    monkeypatch.setitem(main.__globals__, "run_benchmarks", lambda **kwargs: observed.append(kwargs))
    argv = ["main.py", "--no_log_file", "--t_run", "0.1", "--n_run", "1"]
    if mlx:
        argv.append("--mlx")
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert len(observed) == 1
    assert observed[0]["backends"] == expected


def test_missing_external_source_never_fetches(monkeypatch, tmp_path):
    monkeypatch.setenv("VF64_SOURCE_DIR", str(tmp_path / "absent"))

    def unexpected_network(*args, **kwargs):
        raise AssertionError("Simulation source loading attempted network access")

    monkeypatch.setattr(vf64.urllib.request, "urlopen", unexpected_network)
    with pytest.raises(RuntimeError, match="--install"):
        vf64.shader_header()


def test_explicit_install_verifies_files_and_preserves_modified_checkout(monkeypatch, tmp_path):
    root = tmp_path / "provider"
    source, license_text = b"controlled source bytes\n", b"controlled license bytes\n"
    source_hash, license_hash = sha256(source).hexdigest(), sha256(license_text).hexdigest()
    monkeypatch.setenv("VF64_SOURCE_DIR", str(root))
    monkeypatch.setattr(vf64, "SHA256", source_hash)
    monkeypatch.setattr(vf64, "LICENSE_SHA256", license_hash)
    monkeypatch.setattr(vf64, "_FILES", ((vf64.SOURCE_PATH, source_hash), (Path("LICENSE"), license_hash)))
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        return BytesIO(license_text if url.endswith("/LICENSE") else source)

    monkeypatch.setattr(vf64.urllib.request, "urlopen", download)
    receipt = vf64.install()
    assert receipt["status"] == "installed"
    assert len(requests) == 2
    assert all(vf64.REVISION in url for url in requests)
    assert vf64.shader_header() == source.decode()
    vf64.install()
    assert len(requests) == 2
    shader = root / vf64.SOURCE_PATH
    shader.write_bytes(b"user modifications")
    with pytest.raises(ValueError, match="integrity check failed"):
        vf64.install()
    assert shader.read_bytes() == b"user modifications"
    assert len(requests) == 2
    with pytest.raises(ValueError, match="integrity check failed"):
        vf64.shader_header()
