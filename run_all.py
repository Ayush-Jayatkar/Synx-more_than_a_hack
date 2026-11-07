"""Regenerate Synx.csv and validate it: python run_all.py"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES
from synx.spells import predict_labels

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Synx.csv")

if __name__ == "__main__":
    n_frames = GAMES[GAME]["n_video_frames"]
    labels = predict_labels(n_frames)
    assert len(labels) == n_frames and set(labels) <= set("HADX")
    pd.DataFrame({"frame": np.arange(n_frames), "label": labels}).to_csv(OUT, index=False)
    print("Wrote {} rows to {}".format(n_frames, OUT))
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), OUT]))
