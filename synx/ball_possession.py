"""Module C. Nearest player per team and who controls the ball on each frame."""

import numpy as np

from synx.data import AWAY, HOME, NOBODY

# A player controls the ball if it is within reach: about one stride, plus tracking noise
# between the ball and the player's tracked centre.
CONTROL_RADIUS_M = 2.0


def team_distances(ball, players, team):
    """(d_home, d_away): distance in metres from the ball to each team's nearest player. inf if none."""
    d = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    d = np.where(np.isnan(d), np.inf, d)
    return d[:, team == HOME].min(axis=1), d[:, team == AWAY].min(axis=1)


def control(d_home, d_away, radius=CONTROL_RADIUS_M):
    """Per-frame code: HOME/AWAY if that team's nearest player is within reach and nearer. Else NOBODY."""
    out = np.full(len(d_home), NOBODY, dtype=np.int8)
    out[(d_home <= radius) & (d_home <= d_away)] = HOME
    out[(d_away <= radius) & (d_away < d_home)] = AWAY
    return out
