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


_OPTIONAL_RUNTIME_MODULES = {
    "mlx", "mlx_backend", "brian2", "brian2cuda", "torch", "nestgpu", "pygenn",
    "brian2genn", "run_mlx", "run_brian2_cuda", "run_pytorch", "run_nestgpu",
    "run_genn", "run_brian2_genn",
}


@pytest.mark.parametrize("args,backends,durations,trials,rounds", [
    pytest.param(["--mlx"], ["mlx"], [0.1], [1], 1, id="mlx-short-defaults"),
    pytest.param(["--t_run", "1"], ["mlx"], [1.0], [1], 1, id="duration-only"),
    pytest.param(["--n_run", "4"], ["mlx"], [0.1], [4], 1, id="trials-only"),
    pytest.param(["--paper"], ["mlx"], [0.1, 1.0, 10.0, 100.0],
                 [1, 4, 8, 16, 32], 5, id="paper-opt-in"),
    pytest.param(["--mlx", "--paper", "--t_run", "1", "--n_run", "4", "--rounds", "2"],
                 ["mlx"], [1.0], [4], 2, id="paper-explicit-overrides"),
    pytest.param(["--brian2-cpu"], ["cpu"], None, None, 1, id="brian2-cpu"),
    pytest.param(["--brian2cuda-gpu"], ["gpu"], None, None, 1, id="brian2cuda-gpu"),
    pytest.param(["--pytorch"], ["pytorch"], None, None, 1, id="pytorch"),
    pytest.param(["--nestgpu"], ["nestgpu"], None, None, 1, id="nestgpu"),
    pytest.param(["--genn"], ["genn"], None, None, 1, id="genn"),
    pytest.param(["--brian2genn"], ["brian2genn"], None, None, 1, id="brian2genn"),
    pytest.param(["--brian2-cpu", "--pytorch", "--mlx"],
                 ["cpu", "pytorch", "mlx"], None, None, 1, id="combined-backends"),
])
def test_cli_selects_backends_without_loading_optional_runtime(
    monkeypatch, args, backends, durations, trials, rounds,
):
    original_import = builtins.__import__
    cuda_imports = []

    def guarded(name, *args, **kwargs):
        if name == "brian2cuda":
            cuda_imports.append(name)
        elif name.split(".", 1)[0] in _OPTIONAL_RUNTIME_MODULES:
            raise AssertionError(f"CLI parsing imported optional runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "brian2cuda", ModuleType("brian2cuda"))
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "main.py"))
    main = namespace["main"]
    observed = []
    monkeypatch.setitem(main.__globals__, "run_benchmarks", lambda **kwargs: observed.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["main.py", "--no_log_file", *args])
    main()
    invocation, = observed
    assert (invocation["backends"], invocation["t_run_values"],
            invocation["n_run_values"], invocation["rounds"]) == (
        backends, durations, trials, rounds,
    )
    assert cuda_imports == (["brian2cuda"] if "gpu" in backends else [])


def test_no_flags_dispatches_one_short_mlx_trial_and_reports_output(
    monkeypatch, tmp_path, capsys,
):
    import benchmark

    output = tmp_path / "data" / "results"
    output.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "path_res", output)
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "data" / "benchmark-results.csv")
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    runner = ModuleType("run_mlx")
    observed = []
    runner.run_benchmarks = lambda **kwargs: observed.append(kwargs) or []
    monkeypatch.setitem(sys.modules, "run_mlx", runner)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name != "run_mlx" and name.split(".", 1)[0] in _OPTIONAL_RUNTIME_MODULES:
            raise AssertionError(f"Default CLI imported an unexpected runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    entrypoint = str(Path(__file__).parents[1] / "main.py")
    runpy.run_path(entrypoint, run_name="__main__")
    runpy.run_path(entrypoint, run_name="__main__")

    invocation, repeated = observed
    assert (invocation["t_run_values"], invocation["n_run_values"],
            invocation["experiment"]["key"], invocation["round_idx"]) == (
        [0.1], [1], "sugar", 1,
    )
    assert invocation["run_label"] != repeated["run_label"]
    output_message = f"Spike output directory: {output / invocation['run_label']}"
    assert output_message in capsys.readouterr().out
    assert output_message in (output / "benchmarks.log").read_text()


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
