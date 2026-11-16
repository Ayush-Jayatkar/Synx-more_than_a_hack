"""Synthetic-only tests for the central H/A/D probability adapter."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.synx_combined import SynxCombined
from synx.data import AWAY, HOME, ball_speed
from synx.pitch import in_play_window
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES


def make_track(n=40, owner=HOME, ball_x=20.0, away_x=40.0):
    """Construct independent, deterministic synthetic metre-coordinate data."""
    ball = np.zeros((n, 2), dtype=float)
    ball[:, 0] = ball_x
    ball[:, 1] = 30.0
    players = np.full((n, 4, 2), np.nan)
    home_x, away_x_values = (ball_x + 1, away_x), (away_x, away_x + 1)
    if owner == AWAY:
        home_x, away_x_values = (away_x, away_x + 1), (ball_x + 1, ball_x + 2)
    for frame in range(n):
        players[frame, 0] = (home_x[0], 30.0)
        players[frame, 1] = (home_x[1], 31.0)
        players[frame, 2] = (away_x_values[0], 30.0)
        players[frame, 3] = (away_x_values[1], 31.0)
    return {"ball": ball, "players": players,
            "player_speed": np.zeros((n, 4)), "team": np.array([HOME, HOME, AWAY, AWAY]),
            "ball_speed": np.zeros(n)}


def assert_probability_contract(probabilities):
    assert probabilities.ndim == 2 and probabilities.shape[1] == 3
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def copy_track(track):
    return {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in track.items()}


def test_output_contract_normalization_and_nonmutation():
    track = make_track()
    before = {key: value.copy() for key, value in track.items()}
    result = SynxCombined().predict_proba(track)
    assert result.shape == (40, 3)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert np.allclose(result.sum(axis=1), 1.0)
    for key in track:
        assert np.array_equal(track[key], before[key], equal_nan=True)


def test_stable_home_and_away_and_genuine_turnover_evidence():
    model = SynxCombined()
    home = model.predict_proba(make_track(owner=HOME))
    away = model.predict_proba(make_track(owner=AWAY))
    assert np.all(home[:, 0] > home[:, 1])
    assert np.all(away[:, 1] > away[:, 0])
    track = make_track()
    track["ball"][20:, 0] = 40.0
    track["players"][20:, 0, 0] = 60.0
    track["players"][20:, 1, 0] = 61.0
    track["players"][20:, 2, 0] = 41.0
    track["players"][20:, 3, 0] = 42.0
    switched = model.predict_proba(track)
    assert np.all(switched[:20, 0] > switched[:20, 1])
    assert np.all(switched[20:, 1] > switched[20:, 0])

    reverse = make_track(owner=AWAY)
    reverse["ball"][20:, 0] = 20.0
    reverse["players"][20:, 0, 0] = 21.0
    reverse["players"][20:, 1, 0] = 22.0
    reverse["players"][20:, 2, 0] = 60.0
    reverse["players"][20:, 3, 0] = 61.0
    reverse_result = model.predict_proba(reverse)
    assert np.all(reverse_result[20:, 0] > reverse_result[20:, 1])


def test_missing_contested_and_far_loose_ball_are_uncertain_not_dead():
    model = SynxCombined()
    track = make_track()
    track["ball"][4:8] = np.nan
    track["ball"][25:] = np.nan
    track["ball"][10:15] = (100.0, 100.0)
    track["players"][16, 2] = (21.0, 30.0)
    result = model.predict_proba(track)
    assert np.allclose(result[4:8, :2], 0.5)
    assert np.allclose(result[25:, :2], 0.5)
    assert np.allclose(result[10:15, :2], 0.5)
    assert np.all(result[:, 2] == 0.0)


def test_passes_fast_ball_opponent_approach_and_contest():
    model = SynxCombined()
    passing = make_track(12)
    passing["ball"][:, 0] = np.linspace(21.0, 23.0, 12)
    passing["players"][:, 1, 0] = passing["ball"][:, 0] + 0.8
    passing_before = copy_track(passing)
    passed = model.predict_proba(passing)
    assert_probability_contract(passed)
    assert np.all(passed[:, 0] >= 0.75), "controlled Home passing must stay Home-favored"
    assert np.all(passed[:, 0] > passed[:, 1]), "a teammate pass must not look like Away possession"
    for key, value in passing_before.items():
        assert np.array_equal(passing[key], value, equal_nan=True)

    rapid = make_track(12)
    rapid["ball"][:, 0] = np.linspace(21.0, 30.0, 12)
    rapid["ball_speed"] = ball_speed(rapid["ball"])
    fast = model.predict_proba(rapid)
    assert_probability_contract(fast)
    # Fast motion alone is not an opposing-control event: it may reduce
    # confidence as the ball leaves the receiver, but must not force a turnover.
    assert np.all(fast[:, 1] <= 0.55)
    assert np.all(fast[:, 0] >= fast[:, 1])

    approaching = make_track(12)
    approaching["players"][:, 2, 0] = np.linspace(35.0, 25.0, 12)
    pressured = model.predict_proba(approaching)
    assert_probability_contract(pressured)
    assert np.all(pressured[:, 0] > pressured[:, 1]), (
        "an opponent closing distance is pressure, not possession gain"
    )

    contested = make_track(1)
    contested["players"][0, 2] = (19.0, 30.0)
    contested["players"][0, 0] = (21.0, 30.0)
    tied = model.predict_proba(contested)
    clear = model.predict_proba(make_track(1))
    assert_probability_contract(tied)
    assert np.allclose(tied[0, :2], 0.5)
    assert abs(tied[0, 0] - 0.5) < abs(clear[0, 0] - 0.5)

    # Conflicting evidence can be close without being exactly symmetric.
    imperfect = make_track(1)
    imperfect["players"][0, 0] = (21.0, 30.0)
    imperfect["players"][0, 1] = (40.0, 31.0)
    imperfect["players"][0, 2] = (19.4, 30.0)
    imperfect["players"][0, 3] = (60.0, 31.0)
    imperfect_result = model.predict_proba(imperfect)
    assert_probability_contract(imperfect_result)
    assert abs(imperfect_result[0, 0] - 0.5) < abs(clear[0, 0] - 0.5)
    assert abs(imperfect_result[0, 0] - 0.5) < 0.1


def test_goalkeeper_distribution_noise_and_tracking_discontinuity():
    model = SynxCombined()
    keeper = make_track(10)
    keeper["ball"][:, 0] = 3.0
    keeper["players"][:, 0, 0] = 3.5  # home goalkeeper distribution
    distribution = model.predict_proba(keeper)
    assert np.all(distribution[:, 0] > distribution[:, 1])

    noisy = make_track(80)
    rng = np.random.default_rng(19)
    noisy["ball"] += rng.normal(0, 0.03, noisy["ball"].shape)
    noisy["players"] += rng.normal(0, 0.04, noisy["players"].shape)
    jittered = model.predict_proba(noisy)
    assert np.isfinite(jittered).all()
    assert np.allclose(jittered.sum(axis=1), 1.0)

    broken = make_track(8)
    broken["ball"][4:] = (80.0, 80.0)
    discontinuous = model.predict_proba(broken)
    assert np.allclose(discontinuous[4:, :2], 0.5)
    assert np.all(discontinuous[:, 2] == 0.0)


def test_dead_ball_is_separate_from_missing_tracking():
    track = make_track(8)
    track["dead"] = np.zeros(8, dtype=bool)
    track["dead"][3] = True
    track["ball"][5] = np.nan
    result = SynxCombined().predict_proba(track)
    assert np.array_equal(result[3], [0.0, 0.0, 1.0])
    assert np.allclose(result[5], [0.5, 0.5, 0.0])
    assert_probability_contract(result)


def test_pitch_out_of_play_is_dead_but_missing_ball_is_not():
    kickoff = min(KICKOFF_VIDEO_FRAMES.values())
    track = make_track(kickoff + 8)
    track["pitch"] = (105.0, 68.0)
    track["ball"][kickoff + 2] = (108.0, 34.0)
    track["ball"][kickoff + 5] = np.nan
    track["ball_speed"] = ball_speed(track["ball"])
    result = SynxCombined().predict_proba(track)
    assert np.array_equal(result[kickoff + 2], [0.0, 0.0, 1.0])
    assert np.allclose(result[kickoff + 5], [0.5, 0.5, 0.0])
    assert_probability_contract(result)


def test_match_window_boundary_uses_absolute_frames():
    kickoff = min(KICKOFF_VIDEO_FRAMES.values())
    track = make_track(kickoff + 3)
    track["pitch"] = (105.0, 68.0)
    model = SynxCombined()
    whole = model.predict_proba(track)
    prefix_track = {key: (value if key in ("team", "pitch") else value[:kickoff])
                    for key, value in track.items()}
    _, context = model.predict_proba_chunk(prefix_track, None, 0)
    boundary_track = {key: (value if key in ("team", "pitch") else value[kickoff:])
                      for key, value in track.items()}
    boundary, _ = model.predict_proba_chunk(boundary_track, context, kickoff)
    assert np.all(whole[:kickoff, 2] == 1.0)
    assert np.all(whole[kickoff:, 2] == 0.0)
    assert np.allclose(whole[kickoff:], boundary, atol=0.0, rtol=0.0)
    assert_probability_contract(whole)


def test_invalid_dead_mask_cannot_fabricate_dead_ball():
    track = make_track(4)
    track["dead"] = np.array([False, np.nan, "False", False], dtype=object)
    result = SynxCombined().predict_proba(track)
    assert np.allclose(result[:, :2], 0.5)
    assert np.all(result[:, 2] == 0.0)


def test_explicit_dead_evidence_survives_unavailable_possession_features():
    result = SynxCombined().predict_proba({
        "ball": np.full((3, 2), np.nan),
        "dead": np.array([False, True, False]),
    })
    assert np.allclose(result[0], [0.5, 0.5, 0.0])
    assert np.array_equal(result[1], [0.0, 0.0, 1.0])
    assert np.allclose(result[2], [0.5, 0.5, 0.0])


def test_custom_feature_control_radius_is_respected():
    track = make_track(5)
    default = SynxCombined().predict_proba(track)
    narrow = SynxCombined(feature_kwargs={"control_radius": 0.5}).predict_proba(track)
    assert np.all(default[:, 0] > 0.5)
    assert np.max(np.abs(narrow[:, 0] - 0.5)) < np.max(np.abs(default[:, 0] - 0.5))


def test_uncalibrated_evidence_never_returns_certain_team_probability():
    track = make_track(5)
    track["players"][:, 2:, 0] = 1000.0
    result = SynxCombined().predict_proba(track)
    assert np.all(result[:, 0] <= 0.98)
    assert np.all(result[:, 1] >= 0.02)


def test_empty_and_invalid_tracking_are_handled():
    model = SynxCombined()
    assert model.predict_proba({"ball": np.empty((0, 2))}).shape == (0, 3)
    invalid = model.predict_proba({"ball": np.zeros((3, 2))})
    assert invalid.shape == (3, 3)
    assert np.allclose(invalid[:, :2], 0.5)
    assert np.all(invalid[:, 2] == 0.0)
    assert model.predict_proba({}).shape == (0, 3)


def test_chunk_context_matches_whole_track_for_motion_gaps_switches_and_dead():
    kickoff = min(KICKOFF_VIDEO_FRAMES.values())
    n = kickoff + 160
    track = make_track(n)
    frame = np.arange(n, dtype=float)
    relative_frame = frame - kickoff
    track["ball"][:, 0] = 52.5 + 0.006 * relative_frame + 3.0 * np.sin(frame / 41.0)
    track["ball"][:, 1] = 34.0 + 2.0 * np.cos(frame / 57.0)
    # Make possession evidence switch after live play begins.
    switch = kickoff + 70
    track["ball"][switch:, 0] += 20.0
    for slot in range(track["players"].shape[1]):
        phase = slot * 0.61
        for axis in range(2):
            amplitude = 2.5 if axis == 0 else 0.4
            track["players"][:, slot, axis] += (
                amplitude * np.sin(frame / (19.0 + slot) + phase + axis)
            )
    track["players"][:, :2, 0] = (track["ball"][:, 0] + 1.0)[:, None]
    track["players"][:, 2:, 0] = (track["ball"][:, 0] + 18.0)[:, None]
    track["players"][switch:, :2, 0] = (track["ball"][switch:, 0] + 18.0)[:, None]
    track["players"][switch:, 2:, 0] = (track["ball"][switch:, 0] + 1.0)[:, None]
    # Missing tracking followed by visible resumption, including near chunk
    # boundaries for the requested chunk sizes.
    track["ball"][kickoff + 30:kickoff + 47] = np.nan
    track["ball"][kickoff + 111:kickoff + 116] = np.nan
    track["ball"][kickoff + 90:kickoff + 94, 0] = 108.0
    track["pitch"] = (105.0, 68.0)
    track["ball_speed"] = ball_speed(track["ball"])
    model = SynxCombined()
    before = copy_track(track)
    whole = model.predict_proba(track)
    assert_probability_contract(whole)
    assert np.all(whole[:kickoff, 2] == 1.0)
    assert np.all(whole[kickoff:kickoff + 30, 2] == 0.0)
    assert np.all(whole[kickoff + 90:kickoff + 94, 2] == 1.0)
    assert whole[kickoff + 60, 0] > whole[kickoff + 60, 1]
    assert whole[switch + 10, 1] > whole[switch + 10, 0]
    for key, value in before.items():
        assert np.array_equal(track[key], value, equal_nan=True)

    for chunk_size in (1, 25, 100, 512, 4096):
        previous = None
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk = {key: (value if key in ("team", "pitch") else value[start:end])
                     for key, value in track.items()}
            probabilities, previous = model.predict_proba_chunk(
                chunk, previous_features=previous, frame_start=start
            )
            chunks.append(probabilities)
        chunked = np.concatenate(chunks, axis=0)
        assert chunked.shape == whole.shape
        assert np.array_equal(chunked, whole), "chunk output must align exactly by absolute frame"
        assert_probability_contract(chunked)
        assert np.array_equal(chunked[:, 2] == 1.0, whole[:, 2] == 1.0)


def test_pitch_window_chunk_uses_absolute_match_frames():
    kickoff = min(KICKOFF_VIDEO_FRAMES.values())
    start, end = kickoff - 3, kickoff + 4
    whole_mask = in_play_window(end)
    chunk_mask = in_play_window(end - start, start_frame=start)
    assert np.array_equal(whole_mask[start:end], chunk_mask)

    track = make_track(end)
    track["pitch"] = (105.3, 68.0)
    model = SynxCombined()
    whole = model.predict_proba(track)
    prefix, context = model.predict_proba_chunk(
        {key: (value if key in ("team", "pitch") else value[:start])
         for key, value in track.items()},
        previous_features=None, frame_start=0,
    )
    chunk = {key: (value if key in ("team", "pitch") else value[start:end])
             for key, value in track.items()}
    boundary, _ = model.predict_proba_chunk(
        chunk, previous_features=context, frame_start=start
    )
    assert np.array_equal(whole_mask, in_play_window(end))
    assert np.allclose(whole[start:end], boundary)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    print("All probability module tests passed.")
