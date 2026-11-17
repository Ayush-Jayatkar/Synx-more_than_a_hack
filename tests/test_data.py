"""Workstream 1 data and tracking tests. Synthetic files only, never the match data or ground truth.

    python tests/test_data.py
"""

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx import data, missing_ball, tracking_diagnostics  # noqa: E402
from synx.data import AWAY, HOME  # noqa: E402

PITCH = (105.3, 68.0)
N = 199319  # load() always sizes arrays to the tactical video
META = ('<x><Team id="t1"><Name>Home FC</Name></Team><Team id="t2"><Name>Away FC</Name></Team>'
        '<Player teamId="t1"/><Player teamId="t2"/>{}</x>')
SCHEMA_XY = ('<DataFormatSpecifications><DataFormatSpecification><SplitRegister separator=":">'
             '<BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
             '</SplitRegister></DataFormatSpecification></DataFormatSpecifications>')
SCHEMA_XYZ = SCHEMA_XY.replace('<BallChannelRef channelId="y"/>',
                               '<BallChannelRef channelId="y"/><BallChannelRef channelId="z"/>')
LINES = ["1:0.5,0.5,1.0;0.6,0.5,2.0:0.5,0.5,0.4",
         "2:NaN,NaN,NaN;0.6,0.5,2.0:NaN,NaN,NaN",
         "3:0.1,0.2,3.0;0.2,0.2,0.0:0.1,0.1,0.0"]


def make_files(lines=LINES, schema=""):
    folder = tempfile.mkdtemp()
    tactical = os.path.join(folder, "tactical")
    os.makedirs(tactical)
    meta = os.path.join(tactical, "BAR-ATH_tactical_atd.xml")
    raw = os.path.join(tactical, "BAR-ATH_tactical_atd.txt")
    with open(meta, "w") as f:
        f.write(META.format(schema))
    with open(raw, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return folder, meta, raw


def load(folder, **kwargs):
    return data.load(data_dir=folder, cache_dir=os.path.join(folder, "cache"), **kwargs)


def touch_later(path, extra=""):
    """Change a file so the cache must notice: new content and a later mtime."""
    with open(path, "a") as f:
        f.write(extra)
    later = time.time() + 5
    os.utime(path, (later, later))


# ---- indexing, conversion, dimensions -------------------------------------------------------

def test_frame_indexing_and_alignment():
    _, _, raw = make_files(["3:0.1,0.1,0;0.2,0.2,0:0.3,0.3", "1:0.5,0.5,0;0.6,0.6,0:0.5,0.5",
                            "9:0.9,0.9,0;0.9,0.9,0:0.9,0.9"])  # out of order, and one beyond n
    ball, players, speed = data.parse_atd(raw, 2, 4, PITCH)
    assert np.allclose(ball[0], [0.5 * 105.3, 0.5 * 68.0])  # frame_id 1 -> row 0
    assert np.allclose(ball[2], [0.3 * 105.3, 0.3 * 68.0])  # frame_id 3 -> row 2
    assert np.isnan(ball[1]).all() and np.isnan(ball[3]).all()  # never seen; id 9 ignored


def test_coordinate_conversion():
    _, _, raw = make_files(["1:0,0,0;1,1,0:0.25,0.75"])
    ball, players, _ = data.parse_atd(raw, 2, 1, PITCH)
    assert np.allclose(players[0], [[0, 0], [105.3, 68.0]])
    assert np.allclose(ball[0], [26.325, 51.0])


def test_dimensions_and_team():
    folder, meta, _ = make_files()
    track = load(folder)
    slots = len(track["team"])
    assert slots == 2 and list(track["team"]) == [HOME, AWAY]
    assert track["ball"].shape == (N, 2) and track["players"].shape == (N, slots, 2)
    assert track["player_speed"].shape == (N, slots) and track["ball_speed"].shape == (N,)
    assert track["pitch"] == PITCH
    assert np.allclose(track["player_speed"][0], [1.0, 2.0])


# ---- speeds and missing values --------------------------------------------------------------

def test_ball_speed_known_positions():
    ball = np.array([[0, 0], [0.3, 0.4], [0.6, 0.8], [np.nan, np.nan], [1, 1]], dtype=np.float32)
    v = data.ball_speed(ball)
    assert np.isnan(v[0]) and np.allclose(v[1:3], 0.5 * data.FPS)  # 0.5 m per frame = 12.5 m/s
    assert np.isnan(v[3]) and np.isnan(v[4])  # no speed into or out of a missing frame
    assert data.ball_speed(np.zeros((0, 2), dtype=np.float32)).shape == (0,)


def test_missing_observations_stay_nan():
    _, _, raw = make_files()
    ball, players, speed = data.parse_atd(raw, 2, 3, PITCH)
    assert np.isnan(ball[1]).all() and np.isnan(players[1, 0]).all() and np.isnan(speed[1, 0])
    assert np.allclose(players[1, 1], [0.6 * 105.3, 0.5 * 68.0])  # other slot unaffected


def test_long_missing_ball_gaps():
    ball = np.full((200, 2), np.nan, dtype=np.float32)
    ball[:10] = 1.0
    ball[190:] = 2.0  # 180-frame gap (7.2 s)
    long_gap = missing_ball.long_gaps(ball)
    assert long_gap[10:190].all() and not long_gap[:10].any() and not long_gap[190:].any()
    assert np.isnan(missing_ball.interpolate_short_gaps(ball)[100]).all()  # long gap not filled


# ---- ball-speed filter ----------------------------------------------------------------------

def test_filter_removes_noise_spike_and_keeps_raw():
    raw = np.array([10, 10, 60, 10, 10, 10], dtype=np.float32)  # one-frame jitter spike
    before = raw.copy()
    for window in (3, 5):
        filtered = data.filter_ball_speed(raw, window)
        assert np.allclose(filtered, 10.0), window
    assert np.array_equal(raw, before)  # raw array untouched


def test_filter_never_crosses_gaps():
    speed = np.array([5, 5, 5, np.nan, 30, 30, 30], dtype=np.float32)
    filtered = data.filter_ball_speed(speed, 5)
    assert np.isnan(filtered[3])  # nothing invented inside the gap
    assert np.allclose(filtered[:3], 5.0) and np.allclose(filtered[4:], 30.0)  # sides never mix
    long_gap = np.array([7] + [np.nan] * 100 + [9], dtype=np.float32)
    out = data.filter_ball_speed(long_gap, 3)
    assert out[0] == 7 and out[-1] == 9 and np.isnan(out[1:-1]).all()


def test_filter_edges_window_and_units():
    assert np.isnan(data.filter_ball_speed(np.full(4, np.nan, dtype=np.float32), 3)).all()
    assert np.allclose(data.filter_ball_speed(np.array([1, 2, 3], dtype=np.float32), 5), [2, 2, 2])
    for bad in (1, 2, 4, 7):
        try:
            data.filter_ball_speed(np.zeros(5), bad)
            raise AssertionError("window {} accepted".format(bad))
        except ValueError:
            pass


def test_load_adds_filtered_speed_only_when_asked():
    folder, _, _ = make_files()
    assert "ball_speed_filtered" not in load(folder)
    track = load(folder, ball_speed_window=3)
    assert track["ball_speed_filtered"].shape == (N,)
    assert np.array_equal(np.isnan(track["ball_speed_filtered"]), np.isnan(track["ball_speed"]))


# ---- ball height ----------------------------------------------------------------------------

def test_ball_height_absent_when_schema_has_no_z():
    folder, meta, _ = make_files(schema=SCHEMA_XY)
    assert data.ball_channels(meta) == ["x", "y"]
    track = load(folder)
    assert "ball_z" not in track and not data.check(track)["ball_height_available"]
    folder, meta, _ = make_files()  # no schema block at all
    assert data.ball_channels(meta) == [] and "ball_z" not in load(folder)


def test_ball_height_read_when_schema_declares_z():
    folder, meta, raw = make_files(schema=SCHEMA_XYZ)
    assert data.ball_channels(meta) == ["x", "y", "z"]
    ball, _, _, z = data.parse_atd(raw, 2, 3, PITCH, ball_z_index=2)
    assert np.isclose(z[0], 0.4) and np.isnan(z[1]) and z[2] == 0.0  # as reported, NaN kept
    assert np.allclose(ball[0], [0.5 * 105.3, 0.5 * 68.0])  # x, y unaffected by the extra field
    track = load(folder)
    assert np.isclose(track["ball_z"][0], 0.4) and np.isnan(track["ball_z"][1])
    cached = load(folder)
    assert np.array_equal(track["ball_z"], cached["ball_z"], equal_nan=True)


# ---- cache ----------------------------------------------------------------------------------

KEYS = ("ball", "players", "player_speed", "team", "ball_speed")


def test_cache_created_and_matches_raw_load():
    folder, _, _ = make_files()
    first = load(folder)
    assert os.path.isfile(os.path.join(folder, "cache", "BAR-ATH_tracking.npz"))
    cached = load(folder)
    raw_only = load(folder, use_cache=False)
    for key in KEYS:
        assert np.array_equal(first[key], cached[key], equal_nan=True), key
        assert np.array_equal(first[key], raw_only[key], equal_nan=True), key
        assert first[key].dtype == cached[key].dtype, key


def test_cache_rebuilds_when_raw_changes():
    folder, _, raw = make_files()
    assert np.isnan(load(folder)["ball"][3]).all()
    touch_later(raw, "4:0.1,0.1,0.0;0.2,0.2,0.0:0.2,0.2\n")
    assert np.allclose(load(folder)["ball"][3], [0.2 * 105.3, 0.2 * 68.0])


def test_cache_rebuilds_when_metadata_changes():
    folder, meta, _ = make_files()
    assert list(load(folder)["team"]) == [HOME, AWAY]
    with open(meta, "w") as f:  # swap which slot belongs to which team
        f.write(META.format("").replace('<Player teamId="t1"/><Player teamId="t2"/>',
                                        '<Player teamId="t2"/><Player teamId="t1"/>'))
    touch_later(meta)
    assert list(load(folder)["team"]) == [AWAY, HOME]


def test_corrupt_cache_rebuilds():
    folder, _, _ = make_files()
    good = load(folder)
    cache = os.path.join(folder, "cache", "BAR-ATH_tracking.npz")
    for junk in (b"", b"not a zip file", open(cache, "rb").read()[:50]):
        with open(cache, "wb") as f:
            f.write(junk)
        again = load(folder)
        assert np.array_equal(good["ball"], again["ball"], equal_nan=True)
        assert os.path.getsize(cache) > 100  # rewritten with a valid cache
    assert not os.path.exists(cache + ".tmp")


# ---- malformed and empty input --------------------------------------------------------------

def test_empty_tracking_file():
    folder, _, raw = make_files(lines=[])
    ball, players, speed = data.parse_atd(raw, 2, 3, PITCH)
    assert np.isnan(ball).all() and np.isnan(players).all() and np.isnan(speed).all()
    stats = data.check(load(folder))
    assert stats["ball_seen_pct"] == 0.0 and stats["home_slots"] == 1
    assert data.check(load(folder), np.arange(0))["frames"] == 0


def test_malformed_tracking_input():
    _, _, raw = make_files(["", "header line without separators",
                            "1:0.5,0.5,1.0;0.6,0.5,2.0:0.5,0.5"])
    ball, _, _ = data.parse_atd(raw, 2, 2, PITCH)  # non-record lines skipped, as in the baseline
    assert np.allclose(ball[0], [0.5 * 105.3, 0.5 * 68.0])
    for bad in (["1:0.5,abc,1.0;0.6,0.5,2.0:0.5,0.5"],  # unreadable number
                ["x:0.5,0.5,1.0;0.6,0.5,2.0:0.5,0.5"],  # unreadable frame id
                ["1:0.5,0.5,1.0;0.6,0.5:0.5,0.5"]):     # player block does not match slots
        _, _, raw = make_files(bad)
        try:
            data.parse_atd(raw, 2, 2, PITCH)
            raise AssertionError("accepted {}".format(bad))
        except ValueError:
            pass


# ---- duplicate diagnostics ------------------------------------------------------------------

def _players(n=80, slots=4):
    rng = np.random.default_rng(0)
    p = (rng.random((n, slots, 2)) * [105.3, 68.0]).astype(np.float32)
    p[:, :, 0] += np.arange(slots) * 30  # keep slots far apart by default
    return p


def test_duplicate_detected_with_ranges():
    team = np.array([HOME, HOME, AWAY, AWAY])
    players = _players()
    players[10:60, 2] = players[10:60, 0] + [0.05, 0.0]  # away slot 2 sits on home slot 0
    before = players.copy()
    found = tracking_diagnostics.cross_team_duplicates(players, team)
    assert len(found) == 1
    c = found[0]
    assert (c["home_slot"], c["away_slot"], c["start_frame"], c["end_frame"], c["n_frames"]) == (0, 2, 10, 59, 50)
    assert (c["home_team"], c["away_team"]) == (HOME, AWAY)
    assert c["confidence"] == "high" and "50 consecutive frames" in c["reason"]
    assert np.array_equal(players, before)  # never modifies the tracks


def test_duplicate_needs_persistence_threshold_and_opposite_teams():
    team = np.array([HOME, HOME, AWAY, AWAY])
    players = _players(n=120)  # each case below uses its own frames
    players[0:30, 1] = players[0:30, 0]        # same team overlap: not a cross-team duplicate
    players[40:50, 2] = players[40:50, 0]      # only 10 frames: a brief collision, not reported
    assert tracking_diagnostics.cross_team_duplicates(players, team) == []
    assert len(tracking_diagnostics.cross_team_duplicates(players, team, min_frames=10)) == 1
    players[70:110, 3] = players[70:110, 1] + [1.0, 0.0]  # 1 m apart: only with a larger threshold
    assert tracking_diagnostics.cross_team_duplicates(players, team) == []
    found = tracking_diagnostics.cross_team_duplicates(players, team, max_distance_m=1.5)
    assert [(c["home_slot"], c["away_slot"], c["confidence"]) for c in found] == [(1, 3, "medium")]


def test_duplicate_run_broken_by_missing_and_bad_args():
    team = np.array([HOME, AWAY])
    players = _players(slots=2)
    players[0:60, 1] = players[0:60, 0]
    players[30, 1] = np.nan  # a missing frame splits the run into 30 + 29
    found = tracking_diagnostics.cross_team_duplicates(players, team)
    assert [(c["start_frame"], c["end_frame"]) for c in found] == [(0, 29), (31, 59)]
    for kwargs in ({"max_distance_m": 0}, {"min_frames": 0}):
        try:
            tracking_diagnostics.cross_team_duplicates(players, team, **kwargs)
            raise AssertionError(kwargs)
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
