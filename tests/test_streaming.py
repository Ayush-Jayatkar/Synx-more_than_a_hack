"""Synthetic-only tests for independent controls and stateful chunk inference."""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synx import approaches, diagnostics, events, possession_features, spells
from synx.data import AWAY, HOME, NOBODY
from synx.streaming import predict_chunked, simulate_realtime
import run_all


def synthetic_track(n=137):
    time = np.arange(n, dtype=float) / 25.0
    ball = np.column_stack((10 + time, 20 + 0.1 * np.sin(time)))
    players = np.empty((n, 4, 2), dtype=float)
    players[:, 0] = np.column_stack((ball[:, 0] + 0.4, ball[:, 1]))
    players[:, 1] = np.column_stack((ball[:, 0] + 0.7, ball[:, 1] + 0.2))
    players[:, 2] = np.column_stack((ball[:, 0] + 5.0, ball[:, 1]))
    players[:, 3] = np.column_stack((ball[:, 0] + 7.0, ball[:, 1] - 0.2))
    players[40:44, 0] = np.nan  # synthetic tracklet disappearance
    ball[75:78] = np.nan
    speed = np.ones((n, 4), dtype=float)
    speed[40:44, 0] = np.nan
    ball_speed = np.full(n, 1.0)
    ball_speed[75:78] = np.nan
    return {"ball": ball, "players": players, "player_speed": speed,
            "team": np.array([HOME, HOME, AWAY, AWAY]),
            "ball_speed": ball_speed, "pitch": (100.0, 50.0)}


def test_incremental_machine_matches_offline_and_backdates():
    ctrl = np.array([HOME] * 30 + [AWAY] * 20 + [NOBODY] * 2)
    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)
    expected = spells.label_frames(ctrl, dead, seen, switch_frames=5)
    machine = spells.PossessionStateMachine(switch_frames=5)
    chunks = []
    for start in range(0, len(ctrl), 3):
        _, labels = machine.feed(ctrl[start:start + 3], dead[start:start + 3],
                                 seen[start:start + 3])
        chunks.append(labels)
    _, tail = machine.finish()
    chunks.append(tail)
    assert np.array_equal(np.concatenate(chunks), expected)
    assert (expected[30:50] == "A").all()


def test_features_identical_across_chunk_boundaries():
    track = synthetic_track()
    expected = possession_features.build_features(track)
    state = None
    collected = {key: [] for key in expected}
    for start in range(0, len(track["ball"]), 11):
        end = min(start + 11, len(track["ball"]))
        chunk = {key: (value[start:end] if key != "team" else value)
                 for key, value in track.items() if key in
                 ("ball", "players", "player_speed", "team", "ball_speed")}
        result, state = possession_features.build_features(
            chunk, previous=state, return_state=True)
        for key in expected:
            collected[key].append(result[key])
    for key in expected:
        actual = np.concatenate(collected[key], axis=0)
        assert np.array_equal(actual, expected[key], equal_nan=True), key


def test_chunk_sizes_produce_identical_labels():
    track = synthetic_track()
    dead = np.zeros(len(track["ball"]), dtype=bool)
    dead[110:114] = True
    expected = predict_chunked(track, dead, chunk_size=len(dead))
    for size in (1, 2, 13, 32, 128, 512):
        assert np.array_equal(predict_chunked(track, dead, chunk_size=size), expected)


def test_unpaced_live_simulation_matches_offline_and_benchmarks_chunks():
    track = synthetic_track(31)
    dead = np.zeros(31, dtype=bool)
    expected = predict_chunked(track, dead, chunk_size=5)
    live, timing = simulate_realtime(track, dead, pace=False)
    assert np.array_equal(live, expected)
    assert timing["frames"] == len(expected)
    assert timing["paced"] is False
    _, benchmark = diagnostics.benchmark_chunked(track, dead, chunk_size=7)
    assert benchmark["chunk_count"] == 5
    assert benchmark["chunk_processing_ms"]["max"] >= 0


def test_comparison_is_label_free_and_reports_all_approaches():
    track = synthetic_track(50)
    features = possession_features.build_features(track)
    outputs = approaches.compare_controls(
        features, np.zeros(50, dtype=bool), features["ball_seen"])
    assert set(outputs) == {"A_distance", "B_movement", "C_spatial",
                            "D_temporal", "E_combined"}
    report, _, _ = diagnostics.compare_approaches(
        track, np.zeros(50, dtype=bool))
    assert set(report["approaches"]) == set(outputs)
    assert report["pairwise"]
    assert all("accuracy" not in item for item in report["approaches"].values())


def test_event_alignment_skips_unknown_and_malformed_events():
    context = events.align_events([
        {"start_frame": 2, "end_frame": 5, "code": "SET PIECES",
         "tags": {"Team": "home club"}},
        {"start_frame": 4, "end_frame": 8, "code": "BUILD UP",
         "tags": {"Team": "unknown club"}},
        {"start_frame": "bad", "end_frame": 9},
    ], 6, {"home club": HOME})
    assert context["set_piece"].tolist() == [False, False, True, True, True, False]
    assert context["event_present"].sum() == 4
    assert context["event_team"][2] == HOME
    assert context["event_team"][4] == -1
    assert context["skipped_events"] == 1


def test_frame_report_has_aligned_synthetic_rows():
    track = synthetic_track(7)
    features = possession_features.build_features(track)
    control = approaches.combined(features)["control"]
    dead = np.zeros(7, dtype=bool)
    labels = spells.label_frames(control, dead, features["ball_seen"])
    path = os.path.join(tempfile.mkdtemp(), "frame-report.csv")
    diagnostics.write_frame_report(path, labels, features, control, dead)
    with open(path, encoding="utf-8") as handle:
        rows = handle.readlines()
    assert len(rows) == 8
    assert rows[0].startswith("frame,timestamp_s,label,control")


def test_frame_report_vector_fallbacks_are_created_once():
    track = synthetic_track(64)
    features = possession_features.build_features(track)
    control = approaches.combined(features)["control"]
    dead = np.zeros(64, dtype=bool)
    labels = spells.label_frames(control, dead, features["ball_seen"])
    original_full = diagnostics.np.full
    calls = []

    def counted_full(shape, *args, **kwargs):
        if shape == (64, 2):
            calls.append(shape)
        return original_full(shape, *args, **kwargs)

    diagnostics.np.full = counted_full
    try:
        path = os.path.join(tempfile.mkdtemp(), "frame-report-fallback.csv")
        without_vectors = {key: value for key, value in features.items()
                           if key not in ("ball_velocity", "ball_direction")}
        diagnostics.write_frame_report(path, labels, without_vectors, control, dead)
        assert len(calls) == 2  # one (frames, 2) fallback for each optional vector
        with open(path, encoding="utf-8") as handle:
            rows = handle.readlines()
        assert len(rows) == 65

        calls.clear()
        path = os.path.join(tempfile.mkdtemp(), "frame-report-present.csv")
        diagnostics.write_frame_report(path, labels, features, control, dead)
        assert calls == []  # existing arrays do not trigger eagerly evaluated fallbacks
    finally:
        diagnostics.np.full = original_full


def test_synthetic_submission_passes_official_format_validator():
    path = os.path.join(tempfile.mkdtemp(), "synthetic-submission.csv")
    run_all.write_submission(path, np.array(["X", "H", "A", "D"]))
    validator = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "evaluation", "validate_submission.py")
    result = subprocess.run([sys.executable, validator, path, "--rows", "4"],
                            capture_output=True, text=True)
    assert result.returncode == 0, "synthetic submission should validate"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
