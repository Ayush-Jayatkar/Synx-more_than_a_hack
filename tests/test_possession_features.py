
"""Synthetic tests for Workstream 2 possession features."""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from synx import possession_features


def make_track():
    """Create a small synthetic tracking dataset."""

    return {
        "ball": np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [np.nan, np.nan],
            [2.0, 0.0],
        ], dtype=np.float32),

        "players": np.array([
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]],
            [[1.5, 0.0], [3.0, 0.0], [1.0, 2.0]],
            [[2.0, 0.0], [3.0, 0.0], [2.0, 2.0]],
            [[2.5, 0.0], [3.0, 0.0], [2.0, 2.0]],
        ], dtype=np.float32),

        "player_speed": np.array([
            [1.0, 2.0, 3.0],
            [1.5, 2.0, 3.5],
            [2.0, 2.0, 4.0],
            [2.5, 2.0, 4.5],
        ], dtype=np.float32),

        "team": np.array([0, 0, 1]),

        "ball_speed": np.array(
            [np.nan, 25.0, np.nan, np.nan],
            dtype=np.float32,
        ),
    }


def test_feature_keys_and_shapes():
    result = possession_features.build_features(make_track())

    expected = {
        "home_nearest_distance",
        "away_nearest_distance",
        "home_second_distance",
        "away_second_distance",
        "home_nearest_speed",
        "away_nearest_speed",
        "home_players_nearby",
        "away_players_nearby",
        "distance_advantage",
        "ball_speed",
        "ball_direction",
        "ball_velocity",
        "home_approach_speed",
        "away_approach_speed",
        "ball_seen",
    }

    assert set(result) == expected

    for name, values in result.items():
        assert values.shape[0] == 4, name

    assert result["ball_direction"].shape == (4, 2)


def test_nearest_player_distances():
    result = possession_features.build_features(make_track())

    assert np.isclose(result["home_nearest_distance"][0], 1.0)
    assert np.isclose(result["away_nearest_distance"][0], 2.0)


def test_second_nearest_player():
    result = possession_features.build_features(make_track())

    assert np.isclose(result["home_second_distance"][0], 3.0)
    assert np.isnan(result["away_second_distance"][0])


def test_nearest_player_speed():
    result = possession_features.build_features(make_track())

    assert np.isclose(result["home_nearest_speed"][0], 1.0)
    assert np.isclose(result["away_nearest_speed"][0], 3.0)


def test_nearby_player_counts():
    result = possession_features.build_features(make_track())

    assert result["home_players_nearby"][0] == 1
    assert result["away_players_nearby"][0] == 1


def test_distance_advantage():
    result = possession_features.build_features(make_track())

    # Positive means HOME is closer.
    assert np.isclose(result["distance_advantage"][0], 1.0)


def test_ball_direction():
    result = possession_features.build_features(make_track())

    assert np.isnan(result["ball_direction"][0]).all()
    assert np.allclose(result["ball_direction"][1], [1.0, 0.0])
    assert np.isnan(result["ball_direction"][2]).all()
    assert np.isnan(result["ball_direction"][3]).all()


def test_missing_ball():
    result = possession_features.build_features(make_track())

    assert np.isnan(result["home_nearest_distance"][2])
    assert np.isnan(result["away_nearest_distance"][2])
    assert np.isnan(result["distance_advantage"][2])


def test_custom_control_radius():
    result = possession_features.build_features(
        make_track(),
        control_radius=0.5,
    )

    assert result["home_players_nearby"][0] == 0
    assert result["away_players_nearby"][0] == 0


def test_invalid_input():
    track = make_track()
    del track["team"]

    try:
        possession_features.build_features(track)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing input should raise ValueError")




def test_invalid_shapes_and_nonfinite_values():
    track = make_track()
    track["players"][0, 0] = [np.inf, 0]
    track["player_speed"][0, 0] = np.inf
    result = possession_features.build_features(track)
    assert np.isfinite(result["home_nearest_distance"][0])
    assert np.isclose(result["home_nearest_speed"][0], 2.0)
    track["players"] = np.zeros((2, 3, 2))
    try:
        possession_features.build_features(track)
    except ValueError:
        pass
    else:
        raise AssertionError("Frame misalignment must raise ValueError")


def test_empty_frames_and_no_team_slots():
    track = make_track()
    for key in ("ball", "players", "player_speed", "ball_speed"):
        track[key] = track[key][:0]
    result = possession_features.build_features(track)
    assert result["ball_direction"].shape == (0, 2)
    track = make_track()
    track["team"] = np.array([0, 0, 0])
    result = possession_features.build_features(track)
    assert np.isnan(result["away_nearest_distance"]).all()
    assert (result["away_players_nearby"] == 0).all()


def test_large_synthetic_alignment():
    n, slots = 25000, 8
    track = {"ball": np.zeros((n, 2)), "players": np.ones((n, slots, 2)),
             "player_speed": np.ones((n, slots)), "team": np.array([0]*4+[1]*4),
             "ball_speed": np.ones(n)}
    result = possession_features.build_features(track)
    assert all(len(v) == n for v in result.values())

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
