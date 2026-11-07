"""Module D. Fill short ball gaps and flag long ones."""

import numpy as np

FPS = 25
# Over half a second a ball in play moves roughly in a straight line, so linear fill is safe.
INTERP_MAX_GAP = int(0.5 * FPS)
# Ball unseen for longer than a long pass stays in the air (~3 s): treat as dead.
LONG_GAP = int(3.0 * FPS)


def gap_runs(missing):
    """[(start, end)] half-open runs where `missing` is True."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def interpolate_short_gaps(ball, max_gap=INTERP_MAX_GAP):
    """Copy of ball with interior gaps of at most max_gap frames filled linearly."""
    out = ball.copy()
    for start, end in gap_runs(np.isnan(ball[:, 0])):
        if start == 0 or end == len(ball) or end - start > max_gap:
            continue
        t = np.arange(1, end - start + 1)[:, None] / float(end - start + 1)
        out[start:end] = ball[start - 1] + t * (ball[end] - ball[start - 1])
    return out


def long_gaps(ball, min_len=LONG_GAP):
    """True on frames inside a missing-ball run longer than min_len."""
    mask = np.zeros(len(ball), dtype=bool)
    for start, end in gap_runs(np.isnan(ball[:, 0])):
        if end - start > min_len:
            mask[start:end] = True
    return mask
