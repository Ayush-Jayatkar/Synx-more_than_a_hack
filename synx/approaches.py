"""Independent frame-local control approaches for engineering comparisons.

These functions provide alternative evidence models, not accuracy rankings.
They require only synthetic tests for correctness and never consume labels.
Confidence is a transparent score separation measured in metres; it is not a
probability and must not be interpreted as calibrated confidence.
"""
import numpy as np
import inspect
import time

from synx import ball_possession
from synx.data import AWAY, HOME, NOBODY


def _feature(features, name, shape=None):
    if name not in features:
        raise ValueError("Missing feature: " + name)
    value = np.asarray(features[name], dtype=float)
    if value.ndim != 1 or (shape is not None and value.shape != shape):
        raise ValueError("Feature {!r} must be an aligned vector".format(name))
    return value


def distance_based(features, radius=2.0, contest_margin=0.4):
    """Approach A: nearest-team distance plus a fixed contested margin."""
    home = _feature(features, "home_nearest_distance")
    away = _feature(features, "away_nearest_distance", home.shape)
    codes = ball_possession.control(home, away, radius, contest_margin)
    separation = np.abs(home - away)
    confidence = np.maximum(0.0, separation - contest_margin)
    confidence[~(np.isfinite(home) & np.isfinite(away))] = np.nan
    return {"control": codes, "confidence": confidence}


def movement_based(features, radius=2.0, contest_margin=0.4,
                   approach_weight=0.05, ball_uncertainty=0.02,
                   max_uncertainty=1.0):
    """Approach B: distance plus signed nearest-distance closing evidence.

    Positive approach speed means distance closed during the previous frame.
    Ball speed expands the required score margin because a fast-moving ball is
    less likely to indicate settled control. Tracklet swaps make this weak
    evidence; its weight is bounded and explicit.
    """
    home = _feature(features, "home_nearest_distance")
    away = _feature(features, "away_nearest_distance", home.shape)
    hmove = _feature(features, "home_approach_speed", home.shape)
    amove = _feature(features, "away_approach_speed", home.shape)
    speed = _feature(features, "ball_speed", home.shape)
    hscore = home - approach_weight * np.clip(np.nan_to_num(hmove), -5.0, 5.0)
    ascore = away - approach_weight * np.clip(np.nan_to_num(amove), -5.0, 5.0)
    uncertainty = np.minimum(np.nan_to_num(speed, nan=0.0, posinf=50.0), 50.0)
    margin = contest_margin + np.minimum(uncertainty * ball_uncertainty, max_uncertainty)
    codes = np.full(home.shape, NOBODY, dtype=np.int8)
    codes[(home <= radius) & (hscore + margin < ascore)] = HOME
    codes[(away <= radius) & (ascore + margin < hscore)] = AWAY
    confidence = np.abs(hscore - ascore) - margin
    confidence = np.maximum(0.0, confidence)
    confidence[~(np.isfinite(home) & np.isfinite(away))] = np.nan
    return {"control": codes, "confidence": confidence}


def spatial_based(features, radius=2.0, contest_margin=0.4,
                  teammate_penalty=0.25, pressure_margin_per_player=0.1):
    """Approach C: nearest distance adjusted for teammate density and pressure.

    Each additional nearby teammate adds ``teammate_penalty`` metres to its
    team's distance score; nearby players from both teams increase the margin
    needed to call control. Duplicate tracklets may inflate counts, so this
    evidence remains a small adjustment rather than an identity count.
    """
    home = _feature(features, "home_nearest_distance")
    away = _feature(features, "away_nearest_distance", home.shape)
    hc = _feature(features, "home_players_nearby", home.shape)
    ac = _feature(features, "away_players_nearby", home.shape)
    hscore = home + teammate_penalty * np.maximum(hc - 1.0, 0.0)
    ascore = away + teammate_penalty * np.maximum(ac - 1.0, 0.0)
    margin = contest_margin + pressure_margin_per_player * np.maximum(hc + ac - 1.0, 0.0)
    codes = np.full(home.shape, NOBODY, dtype=np.int8)
    codes[(home <= radius) & (hscore + margin < ascore)] = HOME
    codes[(away <= radius) & (ascore + margin < hscore)] = AWAY
    confidence = np.maximum(0.0, np.abs(hscore - ascore) - margin)
    confidence[~(np.isfinite(home) & np.isfinite(away))] = np.nan
    return {"control": codes, "confidence": confidence}


def combined(features, **kwargs):
    """Approach E: current combined evidence model."""
    codes = ball_possession.enhanced_control(features, **kwargs)
    home = _feature(features, "home_nearest_distance")
    away = _feature(features, "away_nearest_distance", home.shape)
    confidence = np.abs(home - away)
    confidence[~(np.isfinite(home) & np.isfinite(away))] = np.nan
    confidence[codes == NOBODY] = 0.0
    return {"control": codes, "confidence": confidence}


_APPROACHES = {
    "A_distance": distance_based,
    "B_movement": movement_based,
    "C_spatial": spatial_based,
    "E_combined": combined,
}


def run_control(name, features, kwargs=None):
    """Run one named stateless approach, filtering unrelated config keys."""
    if name == "D_temporal":
        # D is the temporal state model applied to the common distance evidence.
        name = "A_distance"
    if name not in _APPROACHES:
        raise ValueError("approach must be one of {}".format(sorted(_APPROACHES)))
    function = _APPROACHES[name]
    allowed = inspect.signature(function).parameters
    options = {key: value for key, value in (kwargs or {}).items() if key in allowed}
    return function(features, **options)


def compare_controls(features, dead, ball_seen, switch_frames=13):
    """Run A/B/C/E controls and the D temporal layer for comparison.

    Results are algorithm-behaviour comparisons only. No ground-truth labels,
    scoring metrics, or accuracy claims are used.
    """
    from synx.spells import label_frames

    functions = {"A_distance": distance_based, "B_movement": movement_based,
                 "C_spatial": spatial_based, "E_combined": combined}
    approaches = {}
    for name, function in functions.items():
        started = time.perf_counter()
        approaches[name] = function(features)
        approaches[name]["runtime_s"] = time.perf_counter() - started
    result = {}
    for name, values in approaches.items():
        control = values["control"]
        instant = np.full(control.shape, "X", dtype="<U1")
        instant[control == HOME] = "H"
        instant[control == AWAY] = "A"
        instant[np.asarray(dead, dtype=bool)] = "D"
        temporal_start = time.perf_counter()
        temporal = label_frames(control, dead, ball_seen,
                                switch_frames=switch_frames)
        temporal_runtime = time.perf_counter() - temporal_start
        result[name] = {**values, "instant_labels": instant,
                        "temporal_labels": temporal,
                        "labels": temporal if name == "E_combined" else instant,
                        "temporal_runtime_s": temporal_runtime}
    distance = result["A_distance"]
    result["D_temporal"] = {
        **distance,
        "instant_labels": distance["instant_labels"],
        "temporal_labels": distance["temporal_labels"],
        "labels": distance["temporal_labels"],
        "runtime_s": distance["runtime_s"] + distance["temporal_runtime_s"],
        "temporal_runtime_s": distance["temporal_runtime_s"],
    }
    return result
