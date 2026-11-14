"""Regenerate Synx.csv and validate it: python run_all.py"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from synx import data, possession
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Synx.csv")


def predict(track):
    """Backward-compatible Workstream 2 entry point for loaded tracking."""
    return possession.predict_possession(track)


if __name__ == "__main__":
    labels = predict(data.load())
    assert set(labels) <= set("HADX")
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(OUT, index=False)
    print("Wrote {} rows to {}".format(len(labels), OUT))
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), OUT]))
