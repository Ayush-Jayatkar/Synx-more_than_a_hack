"""Optional nearest-player baseline for synthetic or offline comparisons only.

This baseline is not used by ``run_all.py`` and must not be included in an
official submission without explicit team approval.
"""

import warnings

import numpy as np

from interfaces import AWAY, HOME, PossessionModule, normalize


def _ffill(ball, max_gap):
    n = len(ball)
    seen = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    ok = (last >= 0) & (np.arange(n) - last <= max_gap)
    out = np.full_like(ball, np.nan)
    out[ok] = ball[last[ok]]
    return out


class NearestPlayer(PossessionModule):
    """Soft H/A/D probabilities from nearest-player distances."""

    name = "nearest_player"

    def __init__(self, ffill_s=2.0, fps=25, loose_m=6.0):
        if ffill_s < 0 or fps <= 0 or loose_m < 0:
            raise ValueError("ffill_s/loose_m must be nonnegative and fps positive")
        self.max_gap = int(ffill_s * fps)
        self.loose_m = float(loose_m)

    def predict_proba(self, track):
        if not isinstance(track, dict) or "ball" not in track or "players" not in track or "team" not in track:
            raise ValueError("track must contain ball, players, and team arrays")
        raw_ball = np.asarray(track["ball"], dtype=np.float64)
        players = np.asarray(track["players"], dtype=np.float64)
        team = np.asarray(track["team"])
        if raw_ball.ndim != 2 or raw_ball.shape[1] != 2:
            raise ValueError("track['ball'] must have shape (n_frames, 2)")
        if players.ndim != 3 or players.shape[0] != len(raw_ball) or players.shape[2] != 2:
            raise ValueError("track['players'] must have shape (n_frames, n_players, 2)")
        if team.shape != (players.shape[1],):
            raise ValueError("track['team'] must have one team id per player")
        if not np.isin(team, (HOME, AWAY)).all():
            raise ValueError("track['team'] values must be HOME or AWAY")
        ball = _ffill(raw_ball, self.max_gap)

        distances = np.hypot(
            players[..., 0] - ball[:, None, 0],
            players[..., 1] - ball[:, None, 1],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home_distance = np.nanmin(
                np.where(team[None, :] == HOME, distances, np.nan), axis=1
            )
            away_distance = np.nanmin(
                np.where(team[None, :] == AWAY, distances, np.nan), axis=1
            )

        valid = np.isfinite(home_distance) & np.isfinite(away_distance)
        margin = np.where(valid, away_distance - home_distance, 0.0)
        closest = np.where(valid, np.minimum(home_distance, away_distance), np.inf)
        home_probability = 1.0 / (1.0 + np.exp(-np.clip(margin / 2.0, -40.0, 40.0)))
        dead_probability = 1.0 / (
            1.0 + np.exp(-np.clip((closest - self.loose_m) / 1.5, -40.0, 40.0))
        )
        # Without a visible ball and both teams, this baseline abstains by
        # returning equal team evidence and all dead-ball probability.
        dead_probability[~valid] = 1.0
        return normalize(np.stack((
            (1.0 - dead_probability) * home_probability,
            (1.0 - dead_probability) * (1.0 - home_probability),
            dead_probability,
        ), axis=1))


class NearestPlayerLong(NearestPlayer):
    """Longer ball-gap carry variant, retained for baseline comparisons."""

    name = "nearest_player_long"

    def __init__(self):
        super().__init__(ffill_s=6.0, loose_m=9.0)
