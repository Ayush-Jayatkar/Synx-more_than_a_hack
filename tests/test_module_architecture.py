"""Synthetic-only checks for shared possession-module utilities."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from interfaces import LABELS, PossessionModule, blend_probabilities, check_output, normalize
from modules.nearest_player import NearestPlayer
from modules.registry import DEFAULT_REGISTRY, ModuleRegistry
from modules.synx_combined import SynxCombined
from synx.data import AWAY, HOME


class FixedModule(PossessionModule):
    name = "fixed"

    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def predict_proba(self, track):
        return self.probabilities


def synthetic_track(n=5):
    ball = np.tile((10.0, 5.0), (n, 1))
    players = np.full((n, 4, 2), np.nan)
    players[:, 0] = (11.0, 5.0)
    players[:, 1] = (12.0, 5.0)
    players[:, 2] = (20.0, 5.0)
    players[:, 3] = (21.0, 5.0)
    return {
        "ball": ball,
        "players": players,
        "team": np.array((HOME, HOME, AWAY, AWAY)),
        "player_speed": np.zeros((n, 4)),
        "ball_speed": np.zeros(n),
    }


def test_interface_and_shared_label_order():
    assert isinstance(SynxCombined(), PossessionModule)
    assert LABELS.tolist() == ["H", "A", "D"]


def test_normalize_handles_unnormalized_and_zero_rows():
    scores = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    normalized = normalize(scores)
    assert np.allclose(normalized[0], [0.5, 0.25, 0.25])
    assert np.allclose(normalized[1], [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(normalized.sum(axis=1), 1.0)


@pytest.mark.parametrize("scores", [
    np.array([1.0, 1.0, 1.0]),
    np.array([[1.0, 2.0]]),
    np.array([[1.0, -1.0, 1.0]]),
    np.array([[1.0, np.nan, 1.0]]),
])
def test_normalize_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        normalize(scores)


def test_output_validation_accepts_contract_and_rejects_bad_outputs():
    valid = np.tile([0.7, 0.2, 0.1], (4, 1))
    assert np.array_equal(check_output(valid, 4, "synthetic"), valid)
    invalid = (
        (valid[:3], 4),
        (np.array([[1.0, 0.0, 0.0, 0.0]]), 1),
        (np.array([[0.5, 0.4, 0.0]]), 1),
        (np.array([[1.1, -0.1, 0.0]]), 1),
        (np.array([[np.inf, 0.0, 0.0]]), 1),
    )
    for output, n_frames in invalid:
        with pytest.raises(ValueError):
            check_output(output, n_frames, "bad synthetic")


def test_probability_blending_equal_and_explicit_weights():
    home = np.tile([0.8, 0.1, 0.1], (3, 1))
    away = np.tile([0.1, 0.8, 0.1], (3, 1))
    equal = blend_probabilities([home, away])
    weighted = blend_probabilities([home, away], weights=[3.0, 1.0])
    assert np.allclose(equal, np.tile([0.45, 0.45, 0.1], (3, 1)))
    assert np.allclose(weighted, np.tile([0.625, 0.275, 0.1], (3, 1)))
    assert np.allclose(weighted.sum(axis=1), 1.0)


@pytest.mark.parametrize("outputs, weights", [
    ([], None),
    ([np.ones((2, 3))], [0.0]),
    ([np.ones((2, 3))], [-1.0]),
    ([np.ones((2, 3))], [np.nan]),
    ([np.tile([1.0, 0.0, 0.0], (2, 1)), np.ones((3, 3))], None),
])
def test_probability_blending_rejects_invalid_inputs(outputs, weights):
    with pytest.raises(ValueError):
        blend_probabilities(outputs, weights=weights)


def test_registry_creates_workstream2_and_optional_baseline_modules():
    assert DEFAULT_REGISTRY.names() == ("nearest_player", "synx_combined")
    assert isinstance(DEFAULT_REGISTRY.create("synx_combined"), SynxCombined)
    assert isinstance(DEFAULT_REGISTRY.create("nearest_player"), NearestPlayer)

    registry = ModuleRegistry()
    registry.register("fixed", FixedModule)
    assert isinstance(registry.create("fixed"), FixedModule)
    with pytest.raises(ValueError):
        registry.register("fixed", FixedModule)
    with pytest.raises(KeyError):
        registry.create("missing")


def test_nearest_player_baseline_returns_probabilities_without_pipeline_wiring():
    track = synthetic_track()
    output = NearestPlayer().predict_proba(track)
    assert output.shape == (5, 3)
    assert np.all(output[:, 0] > output[:, 1])
    assert np.allclose(output.sum(axis=1), 1.0)

    track["ball"][2:] = np.nan
    missing = NearestPlayer(ffill_s=0).predict_proba(track)
    assert np.allclose(missing[2:], [0.0, 0.0, 1.0])


def test_synx_combined_still_works_through_common_contract():
    track = synthetic_track()
    output = SynxCombined().predict_proba(track)
    assert check_output(output, len(track["ball"]), "synx_combined").shape == (5, 3)
