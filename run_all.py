"""Regenerate Synx.csv and validate it: python run_all.py"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from synx import ball_possession, data, missing_ball, pitch, spells

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Synx.csv")


def predict(track):
    """Per-frame labels from the loaded tracking dict."""
    n = len(track["ball"])
    ball = missing_ball.interpolate_short_gaps(track["ball"])
    seen = ~np.isnan(ball[:, 0])
    ctrl = ball_possession.control(*ball_possession.team_distances(ball, track["players"], track["team"]))
    dead = (~pitch.in_play_window(n)) | pitch.ball_out_of_pitch(ball, track["pitch"]) \
        | missing_ball.long_gaps(ball)
    return spells.label_frames(ctrl, dead, seen)


if __name__ == "__main__":
    labels = predict(data.load())
    assert set(labels) <= set("HADX")
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(OUT, index=False)
    print("Wrote {} rows to {}".format(len(labels), OUT))
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), OUT]))
