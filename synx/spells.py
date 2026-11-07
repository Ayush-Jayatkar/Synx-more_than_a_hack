"""Module E. Dead-ball logic, hysteresis, minimum spell length, X policy.

Function signatures: agree in Discord before implementing.
"""

import numpy as np


def predict_labels(n_frames):
    """Return one label (H/A/D/X) per tactical video frame."""
    # shortcut: abstains on every frame so the pipeline runs end to end; replace with real logic.
    return np.full(n_frames, "X", dtype="<U1")
