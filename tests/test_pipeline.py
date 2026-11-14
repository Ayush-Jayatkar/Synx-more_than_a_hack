
"""Synthetic self-checks only. Never use match files or ground truth."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from synx import ball_possession, data, missing_ball, pitch, spells
from synx.data import AWAY, HOME, NOBODY
from synx import possession_features, possession

PITCH = (100.0, 50.0)


def _fake_files(folder):
    tactical = os.path.join(folder, "tactical")
    os.makedirs(tactical)

    meta = os.path.join(tactical, "BAR-ATH_tactical_atd.xml")
    raw = os.path.join(tactical, "BAR-ATH_tactical_atd.txt")

    with open(meta, "w") as f:
        f.write(
            '<x><Team id="t1"><Name>Home FC</Name></Team>'
            '<Team id="t2"><Name>Away FC</Name></Team>'
            '<Player teamId="t1"/><Player teamId="t2"/></x>'
        )

    with open(raw, "w") as f:
        f.write("1:0.5,0.5,1.0;0.6,0.5,2.0:0.51,0.5,0.0\n")
        f.write("2:NaN,NaN,NaN;0.6,0.5,2.0:NaN,NaN,NaN\n")

    return meta, raw


def test_parse_and_team():
    meta, raw = _fake_files(tempfile.mkdtemp())

    team = data.team_of_each_track(meta)
    assert list(team) == [HOME, AWAY]

    ball, players, speed = data.parse_atd(raw, 2, 3, PITCH)

    assert np.allclose(ball[0], [51.0, 25.0])
    assert np.isnan(ball[1]).all() and np.isnan(ball[2]).all()
    assert np.allclose(players[0, 1], [60.0, 25.0])
    assert np.isnan(players[1, 0]).all()
    assert np.allclose(speed[0], [1.0, 2.0])
    assert np.isnan(speed[1, 0]) and speed[1, 1] == 2.0


def test_ball_speed():
    ball = np.array(
        [[0, 0], [0.4, 0.3], [np.nan, np.nan], [1, 1]],
        dtype=np.float32,
    )

    v = data.ball_speed(ball)

    assert np.isnan(v[0])
    assert np.isclose(v[1], 0.5 * data.FPS)
    assert np.isnan(v[2]) and np.isnan(v[3])


def test_load_cache_and_check():
    folder = tempfile.mkdtemp()
    meta, raw = _fake_files(folder)
    cache_dir = os.path.join(folder, "cache")

    first = data.load(data_dir=folder, cache_dir=cache_dir)

    assert os.path.isfile(
        os.path.join(cache_dir, "BAR-ATH_tracking.npz")
    )

    second = data.load(data_dir=folder, cache_dir=cache_dir)

    for name in ("ball", "players", "player_speed", "team", "ball_speed"):
        assert np.array_equal(
            first[name], second[name], equal_nan=True
        ), name

    with open(raw, "a") as f:
        f.write("3:0.1,0.1,0.0;0.2,0.2,0.0:0.1,0.1\n")

    assert not np.isnan(
        data.load(data_dir=folder, cache_dir=cache_dir)["ball"][2]
    ).any()

    stats = data.check(first, np.arange(3))

    assert stats["frames"] == 3
    assert stats["home_slots"] == 1
    assert stats["away_slots"] == 1
    assert np.isclose(stats["ball_seen_pct"], 33.3)
    assert stats["longest_ball_gap_s"] == 0.1


def test_control():
    ball = np.array(
        [[50, 25], [70, 25], [10, 10]],
        dtype=np.float32,
    )

    players = np.array(
        [
            [[51, 25], [55, 25]],
            [[60, 25], [69, 25]],
            [[30, 30], [40, 40]],
        ],
        dtype=np.float32,
    )

    ctrl = ball_possession.control(
        *ball_possession.team_distances(
            ball, players, np.array([HOME, AWAY])
        )
    )

    assert list(ctrl) == [HOME, AWAY, NOBODY]


def test_missing_ball():
    ball = np.array(
        [[0, 0], [np.nan, np.nan], [np.nan, np.nan], [3, 3]]
        + [[np.nan, np.nan]] * 100,
        dtype=np.float32,
    )

    filled = missing_ball.interpolate_short_gaps(ball)

    assert np.allclose(filled[1], [1, 1])
    assert np.allclose(filled[2], [2, 2])
    assert missing_ball.long_gaps(ball)[4:].all()
    assert not missing_ball.long_gaps(ball)[:4].any()


def test_pitch():
    ball = np.array(
        [[50, 25], [-1, 25], [50, 51], [np.nan, np.nan]],
        dtype=np.float32,
    )

    assert list(pitch.ball_out_of_pitch(ball, PITCH)) == [
        False, True, True, False
    ]

    window = pitch.in_play_window(200000)

    assert not window[11184]
    assert window[11185]
    assert not window[100000]
    assert window[113742]


def test_spells():
    n = 200
    s = spells.SWITCH_CONTROL_FRAMES

    ctrl = np.full(n, NOBODY, dtype=np.int8)
    ctrl[10:60] = HOME
    ctrl[80:83] = AWAY
    ctrl[120:120 + s] = AWAY

    dead = np.zeros(n, dtype=bool)
    dead[180:] = True

    labels = spells.label_frames(
        ctrl, dead, np.ones(n, dtype=bool)
    )

    assert (labels[:10] == "X").all()
    assert (labels[10:120] == "H").all()
    assert (labels[120:180] == "A").all()
    assert (labels[180:] == "D").all()

    short = np.array(list("DDHHHDD"))

    assert "".join(spells.drop_short_spells(short)) == "DDXXXDD"


def test_contested_control():
    home = np.array([1.0, 1.0, 3.0, 1.0])
    away = np.array([1.1, 2.0, 1.0, 1.6])

    result = ball_possession.control(
        home, away, contest_margin=0.4
    )

    assert list(result) == [
        NOBODY, HOME, AWAY, HOME
    ]


def test_initial_possession():
    s = spells.SWITCH_CONTROL_FRAMES

    ctrl = np.array([HOME] * s)
    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels == "H").all()


def test_short_genuine_possession():
    s = spells.SWITCH_CONTROL_FRAMES

    ctrl = np.array([HOME] * 30 + [AWAY] * s)
    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels[:30] == "H").all()
    assert (labels[30:] == "A").all()


def test_brief_opposing_touch():
    ctrl = np.array(
        [HOME] * 30 + [AWAY] * 3 + [HOME] * 10
    )

    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels == "H").all()


def test_uncertain_frames_interrupt_switch():
    s = spells.SWITCH_CONTROL_FRAMES

    ctrl = np.array(
        [HOME] * 30
        + [AWAY] * 5
        + [NOBODY] * 3
        + [AWAY] * 5
    )

    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels == "H").all()


def test_missing_ball_preserves_possession():
    ctrl = np.array([HOME] * 30 + [NOBODY] * 5)
    dead = np.zeros(len(ctrl), dtype=bool)
    seen = np.ones(len(ctrl), dtype=bool)

    seen[30:] = False

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels == "H").all()


def test_dead_ball_resets_possession():
    ctrl = np.array(
        [HOME] * 30 + [AWAY] * 30
    )

    dead = np.zeros(len(ctrl), dtype=bool)
    dead[30:35] = True

    seen = np.ones(len(ctrl), dtype=bool)

    labels = spells.label_frames(ctrl, dead, seen)

    assert (labels[:30] == "H").all()
    assert (labels[30:35] == "D").all()
    assert (labels[35:] == "A").all()



def test_enhanced_control_and_prediction_alignment():
    track = {
        "ball": np.array([[0., 0.]] * 30),
        "players": np.array([[[0.5, 0.], [5., 0.]]] * 30),
        "player_speed": np.zeros((30, 2)),
        "team": np.array([HOME, AWAY]),
        "ball_speed": np.zeros(30),
        "pitch": PITCH,
    }
    labels, features = possession.predict_possession(
        track, dead=np.zeros(30, dtype=bool), return_features=True)
    assert labels.shape == (30,)
    assert (labels == "H").all()
    assert len(features["control"]) == 30


def test_missing_ball_is_not_dead():
    track = {
        "ball": np.full((4, 2), np.nan),
        "players": np.zeros((4, 2, 2)),
        "player_speed": np.zeros((4, 2)),
        "team": np.array([HOME, AWAY]),
        "ball_speed": np.full(4, np.nan),
        "pitch": PITCH,
    }
    labels, diagnostic = possession.predict_possession(
        track, dead=np.zeros(4, dtype=bool), return_features=True)
    assert (labels == "X").all()
    assert not diagnostic["dead"].any()


def test_empty_control_inputs():
    assert ball_possession.control(np.array([]), np.array([])).shape == (0,)
    h, a = ball_possession.team_distances(
        np.empty((0, 2)), np.empty((0, 0, 2)), np.array([]))
    assert h.shape == a.shape == (0,)

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
