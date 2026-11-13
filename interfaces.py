"""
interfaces.py - the contract every Synx module follows.

A module receives the `track` dict returned by synx.data.load() and returns
an array of shape (n_frames, 3): probabilities for [H, A, D] on EVERY video frame
(row i = tactical video frame i = submission row i). Rows sum to 1, no NaN.

Modules return probabilities, NOT labels. Smoothing, minimum spell length and
abstaining (X) are decided in ONE place: run_all.py.
"""
import numpy as np

HOME, AWAY = 0, 1                      # same values as synx.data
LABELS = np.array(["H", "A", "D"])


class PossessionModule:
    name = "base"

    def predict_proba(self, track: dict) -> np.ndarray:
        raise NotImplementedError


def normalize(p):
    """Clip negatives, make rows sum to 1 (uniform where a row is all zero)."""
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64)), 0, None)
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.where(s == 0, 1, s), 1.0 / 3.0)


def check_output(p, n_frames, who="?"):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n_frames, 3):
        raise ValueError(f"[{who}] expected shape ({n_frames}, 3), got {p.shape}")
    if np.isnan(p).any():
        raise ValueError(f"[{who}] NaN in probabilities")
    if (p < -1e-9).any():
        raise ValueError(f"[{who}] negative probability")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"[{who}] rows do not sum to 1")
    return p    