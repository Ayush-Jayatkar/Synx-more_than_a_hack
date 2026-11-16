"""Workstream 1 diagnostics: suspicious opposite-team tracklets that sit on top of each other.

The tracker sometimes follows one person with two slots. When those slots belong to different
teams, nearest-player features can see a fake opponent next to the ball. This module only REPORTS
candidate pairs; it never deletes or edits tracks. Downstream modules may use the report or ignore it.

    python -m synx.tracking_diagnostics      # writes outputs/workstream1/duplicate_candidates.csv (local, gitignored)

Defaults come from body size, not from scores: two real players' tracked centres stay about a
shoulder-width (~0.4-0.5 m) apart even in a tight duel, so centres within 0.3 m for a full
second are very unlikely to be two different people.
"""

import os

import numpy as np

from synx.data import AWAY, FPS, HOME

MAX_DISTANCE_M = 0.3
MIN_FRAMES = FPS  # 1 s


def _runs(mask):
    """[(start, end)] half-open runs where `mask` is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def cross_team_duplicates(players, team, max_distance_m=MAX_DISTANCE_M, min_frames=MIN_FRAMES):
    """Candidate duplicate pairs: one HOME and one AWAY slot within max_distance_m for >= min_frames
    consecutive frames (a frame where either slot is missing breaks the run).

    Returns a list of dicts sorted by start frame, each with home_slot, away_slot, home_team,
    away_team, start_frame, end_frame (inclusive), n_frames, median_distance_m, confidence
    ("high" if the run lasts at least twice min_frames and the median distance is at most half
    the threshold, else "medium") and a plain-language reason. Inputs are not modified.
    """
    if max_distance_m <= 0 or min_frames < 1:
        raise ValueError("max_distance_m must be > 0 and min_frames >= 1")
    players = np.asarray(players)
    team = np.asarray(team)
    if players.ndim != 3 or players.shape[2] != 2 or team.shape != (players.shape[1],):
        raise ValueError("expected players (frames, slots, 2) and team (slots,)")
    home_slots = np.flatnonzero(team == HOME)
    away_slots = np.flatnonzero(team == AWAY)
    found = []
    for h in home_slots:
        delta = players[:, away_slots, :] - players[:, h:h + 1, :]
        dist = np.hypot(delta[..., 0], delta[..., 1])  # NaN where either slot is missing
        with np.errstate(invalid="ignore"):
            close = dist <= max_distance_m
        for col, a in enumerate(away_slots):
            for start, end in _runs(close[:, col]):
                n = int(end - start)
                if n < min_frames:
                    continue
                median = float(np.median(dist[start:end, col]))
                high = n >= 2 * min_frames and median <= max_distance_m / 2
                found.append({
                    "home_slot": int(h), "away_slot": int(a),
                    "home_team": int(team[h]), "away_team": int(team[a]),
                    "start_frame": int(start), "end_frame": int(end - 1), "n_frames": n,
                    "median_distance_m": round(median, 3),
                    "confidence": "high" if high else "medium",
                    "reason": "opposite-team slots within {} m for {} consecutive frames ({:.1f} s)".format(
                        max_distance_m, n, n / float(FPS)),
                })
    return sorted(found, key=lambda c: (c["start_frame"], c["home_slot"], c["away_slot"]))


if __name__ == "__main__":
    import pandas as pd
    from synx import data
    out = os.path.join(data.ROOT, "outputs", "workstream1", "duplicate_candidates.csv")
    track = data.load()
    candidates = cross_team_duplicates(track["players"], track["team"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(candidates).to_csv(out, index=False)
    print("{} candidate pairs written to {} (local only, do not share)".format(len(candidates), out))
