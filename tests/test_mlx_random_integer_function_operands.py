"""Integer-valued calls remain live across random-expression operators."""

import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_random_integer_function_operands(tmp_path, monkeypatch, target):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    def build(mode):
        fixture = build_original_api_fixture(
            tmp_path / str(mode=='native'),
            parameters=dict(
                eqs="v:volt\ng:volt\nrfc:second\nrem:integer\nquo:integer\nden:integer",
                eq_th="rand()<.75",
                eq_rst=(
                    "rem=(int(randn()*3)+poisson(2))%den;"
                    "quo=(int(rand()*7)+int(clip(randn(),-2,2)))//den;"
                    "v=v_rst+(quo+randn())*mV;g=0*mV"
                ),
                t_rfc=0*b.ms, t_dly=.2*b.ms,
                r_poi=3700*b.Hz, r_poi2=7200*b.Hz,
            ),
            exc=(0, 0), exc2=(2,), silenced=(),
            initial_state={"den": [-3, 2, -3, 2]},
        )
        return fixture

    def exercise(fixture, mode):
        for duration in (.2, 0., .3):
            yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=943,target=target)


@pytest.mark.parametrize("packed", [False, True])
def test_numpy_random_integer_cast_selected_prefix(tmp_path, monkeypatch, packed, size=49):
    import brian2 as b
    import numpy as np
    import original_api_reference as reference
    from original_api_protocol import check_original_api_runs, observe_run
    monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(size))
    # Selected reset lanes have a vector prefix and a scalar tail. Large
    # finite values distinguish their actual CPU conversion instructions.
    values = np.resize([4294967297., -4294967297., 2.**63, -2.**64,
                        1.5, -1.5, 0.], size)

    def build(mode):
        fixture = reference.build_original_api_fixture(
            tmp_path/str(mode=='native'),
            parameters=dict(
                eqs=("v:volt\ng:volt\nrfc:second\nx:1\na:integer" +
                     ("".join(f"\npad{n}:1" for n in range(24)) if packed else "")),
                eq_th="rand()<int(i%4!=1)",
                eq_rst="a=int(x+rand());v=v_rst;g=" + ("rand()*mV" if packed else "0*mV"),
                t_rfc=0*b.ms, t_dly=.2*b.ms,
                r_poi=3700*b.Hz, r_poi2=7200*b.Hz,
            ),
            exc=(0, 0), exc2=(2,), silenced=(),
            initial_state={"x": values},
        )
        return fixture

    def exercise(fixture, mode):
        for duration in (.1, 0., .2):
            yield observe_run(fixture,duration,rng=True)
    check_original_api_runs(build,exercise,monkeypatch,seed=947,target='numpy')


def test_numpy_ranked_reset_large_prefix_budget(tmp_path, monkeypatch):
    test_numpy_random_integer_cast_selected_prefix(tmp_path, monkeypatch, False, size=65536)
