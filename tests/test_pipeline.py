"""Self-checks on synthetic data only (never the match files): python tests/test_pipeline.py"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx import ball_possession, data, missing_ball, pitch, spells  # noqa: E402
from synx.data import AWAY, HOME, NOBODY  # noqa: E402

PITCH = (100.0, 50.0)


def test_parse_and_team():
    folder = tempfile.mkdtemp()
    meta, raw = os.path.join(folder, "m.xml"), os.path.join(folder, "r.txt")
    with open(meta, "w") as f:
        f.write('<x><Team id="t1"><Name>Home FC</Name></Team><Team id="t2"><Name>Away FC</Name></Team>'
                '<Player teamId="t1"/><Player teamId="t2"/></x>')
    with open(raw, "w") as f:
        f.write("1:0.5,0.5,1.0;0.6,0.5,2.0:0.51,0.5,0.0\n")
        f.write("2:NaN,NaN,NaN;0.6,0.5,2.0:NaN,NaN,NaN\n")
    team = data.team_of_each_track(meta)
    assert list(team) == [HOME, AWAY]
    ball, players = data.parse_atd(raw, 2, 3, PITCH)
    assert np.allclose(ball[0], [51.0, 25.0]) and np.isnan(ball[1]).all() and np.isnan(ball[2]).all()
    assert np.allclose(players[0, 1], [60.0, 25.0]) and np.isnan(players[1, 0]).all()


def test_control():
    ball = np.array([[50, 25], [70, 25], [10, 10]], dtype=np.float32)
    players = np.array([[[51, 25], [55, 25]], [[60, 25], [69, 25]], [[30, 30], [40, 40]]], dtype=np.float32)
    ctrl = ball_possession.control(*ball_possession.team_distances(ball, players, np.array([HOME, AWAY])))
    assert list(ctrl) == [HOME, AWAY, NOBODY]


def test_missing_ball():
    ball = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [3, 3]] + [[np.nan, np.nan]] * 100,
                    dtype=np.float32)
    filled = missing_ball.interpolate_short_gaps(ball)
    assert np.allclose(filled[1], [1, 1]) and np.allclose(filled[2], [2, 2])
    assert missing_ball.long_gaps(ball)[4:].all() and not missing_ball.long_gaps(ball)[:4].any()


def test_pitch():
    ball = np.array([[50, 25], [-1, 25], [50, 51], [np.nan, np.nan]], dtype=np.float32)
    assert list(pitch.ball_out_of_pitch(ball, PITCH)) == [False, True, True, False]
    window = pitch.in_play_window(200000)
    assert not window[11184] and window[11185] and not window[100000] and window[113742]


def test_spells():
    n, s = 200, spells.SWITCH_CONTROL_FRAMES
    ctrl = np.full(n, NOBODY, dtype=np.int8)
    ctrl[10:60] = HOME
    ctrl[80:83] = AWAY            # deflection: ignored
    ctrl[120:120 + s] = AWAY      # sustained: switch, back-dated to frame 120
    dead = np.zeros(n, dtype=bool)
    dead[180:] = True
    labels = spells.label_frames(ctrl, dead, np.ones(n, dtype=bool))
    assert (labels[:10] == "X").all()
    assert (labels[10:120] == "H").all()
    assert (labels[120:180] == "A").all()
    assert (labels[180:] == "D").all()
    short = np.array(list("DDHHHDD"))
    assert "".join(spells.drop_short_spells(short)) == "DDXXXDD"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
