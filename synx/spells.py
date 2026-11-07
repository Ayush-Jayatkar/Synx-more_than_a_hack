"""Module E. Turn per-frame control into possession spells: D logic, hysteresis, min spell, X policy."""

import numpy as np

from synx.data import AWAY, HOME, NOBODY

FPS = 25
# A touch or deflection lasts a few frames; real control means several touches over ~0.5 s.
SWITCH_CONTROL_FRAMES = int(0.5 * FPS)
# A possession shorter than 1 s is too short to be a real spell: abstain on it.
MIN_SPELL_FRAMES = FPS

LETTER = {HOME: "H", AWAY: "A", None: "X"}


def label_frames(ctrl, dead, ball_seen):
    """Per-frame H/A/D/X.

    ctrl: control code per frame. dead: True where play is stopped. ball_seen: ball position known.
    Possession switches only after the other team controls the ball for SWITCH_CONTROL_FRAMES
    frames, with no gap longer than that between them, and the switch is back-dated to its first
    control frame. Live ball with no
    established possession (e.g. right after a restart) is X.
    """
    n = len(ctrl)
    labels = np.full(n, "X", dtype="<U1")
    state = pending = None
    pending_start = last = count = 0

    for i in range(n):
        if dead[i]:
            labels[i] = "D"
            state = pending = None
            continue
        c = int(ctrl[i]) if ball_seen[i] else NOBODY
        if c == NOBODY or c == state:
            if c == state:
                pending = None
            labels[i] = LETTER[state]
            continue
        if c != pending or i - last > SWITCH_CONTROL_FRAMES:
            pending, pending_start, count = c, i, 0
        count += 1
        last = i
        if count >= SWITCH_CONTROL_FRAMES:
            labels[pending_start:i + 1] = LETTER[c]
            state, pending = c, None
        else:
            labels[i] = LETTER[state]
    return drop_short_spells(labels)


def drop_short_spells(labels, min_len=MIN_SPELL_FRAMES):
    """Replace H/A runs shorter than min_len with X."""
    out = labels.copy()
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    for start, end in zip(starts, ends):
        if labels[start] in ("H", "A") and end - start < min_len:
            out[start:end] = "X"
    return out
