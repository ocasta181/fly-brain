"""P9 adapter wiring on three identifiers, without running a network."""

import numpy as np
import pytest

import mlx_engine
from mlx_backend.experiments import EXPERIMENTS, P9_IDS
from mlx_backend.models import Connectome
from mlx_backend.stimuli import SeededEvents


def tiny_p9_graph():
    # Deliberately reverse P9 rows and add an unrelated identifier: target
    # mapping must follow the dataset identifiers, not preset row positions.
    return Connectome(np.array([P9_IDS[1], 123, P9_IDS[0]], np.int64),
        np.array([], np.int32), np.array([], np.int32),
        np.array([], np.int32), np.array([], np.float64))


def test_p9_targets_rates_and_trial_streams_without_simulation():
    spec = EXPERIMENTS["p9"]
    inputs, silenced = mlx_engine.stimuli(tiny_p9_graph(), spec, 67, 3, 731,
                                        stream=True)
    assert silenced == ()
    assert spec.activation_classes == (1, 1)
    assert spec.rates_hz == (100., 100.)
    for trial, stimulus in enumerate(inputs):
        assert stimulus.targets == (2, 0)
        assert stimulus.rates_hz == (100., 100.)
        assert stimulus.trial_indices == (trial,)
        assert stimulus.generator_code == 1 and stimulus.seed == 731
        assert stimulus.amplitudes_mv is None  # Use the resolved model weight.
        expected = SeededEvents(67, (100., 100.), (trial,), 731, 1)
        np.testing.assert_array_equal(stimulus.events.chunk(0, 67),
                                      expected.chunk(0, 67))


@pytest.mark.parametrize("rates", [(0., 10000.), (10000., 0.)])
def test_p9_overlapping_classes_keep_independent_rates(rates):
    spec = mlx_engine.experiment_spec(dict(key="p9", neu_exc=[P9_IDS[0]],
        neu_exc2=[P9_IDS[0], P9_IDS[1]], stim_rate=rates[0],
        stim_rate2=rates[1], neu_slnc=[P9_IDS[1]]))
    inputs, silenced = mlx_engine.stimuli(tiny_p9_graph(), spec, 17, 1, 9,
                                        stream=False)
    stimulus = inputs[0]
    assert spec.activation_classes == (1, 2, 2)
    assert stimulus.targets == (2, 2, 0) and silenced == (0,)
    assert stimulus.rates_hz == (rates[0], rates[1], rates[1])
    np.testing.assert_array_equal(stimulus.events[0],
        np.broadcast_to(np.array([rates[0] > 0, rates[1] > 0,
                                  rates[1] > 0], np.uint8), (17, 3)))
