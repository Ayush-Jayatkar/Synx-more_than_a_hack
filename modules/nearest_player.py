"""Optional nearest-player baseline for synthetic or offline comparisons only.

This baseline is not used by ``run_all.py`` and must not be included in an
official submission without explicit team approval.
"""

import numpy as np

from interfaces import AWAY, HOME, PossessionModule, normalize


class NearestPlayer(PossessionModule):
    """Optional distance-only H/A baseline; D requires an explicit dead mask.

    Missing or unusable tracking yields equal H/A evidence and zero D. The
    baseline does not infer dead ball from distance or missing observations.
    """

    name = "nearest_player"

    def __init__(self, ffill_s=2.0, fps=25, loose_m=6.0, temperature_m=2.0):
        # Keep the former constructor arguments accepted for callers, while no
        # longer using them to turn missing or distant tracking into D.
        legacy = np.asarray((ffill_s, fps, loose_m), dtype=float)
        if (not np.isfinite(legacy).all() or ffill_s < 0 or fps <= 0
                or loose_m < 0):
            raise ValueError("ffill_s/loose_m must be nonnegative and fps positive")
        if not np.isfinite(temperature_m) or temperature_m <= 0:
            raise ValueError("temperature_m must be finite and positive")
        self.temperature_m = float(temperature_m)

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
        if np.isinf(raw_ball).any():
            raise ValueError("ball coordinates must be finite or NaN")
        if np.isinf(players).any():
            raise ValueError("player coordinates must be finite or NaN")
        # A partially observed coordinate is not a usable position.
        ball = raw_ball.copy()
        ball[~np.isfinite(ball).all(axis=1)] = np.nan
        players = players.copy()
        players[~np.isfinite(players).all(axis=2)] = np.nan

        distances = np.hypot(
            players[..., 0] - ball[:, None, 0],
            players[..., 1] - ball[:, None, 1],
        )
        home_distance = np.min(
            np.where(team[None, :] == HOME, distances, np.inf), axis=1,
            initial=np.inf,
        )
        away_distance = np.min(
            np.where(team[None, :] == AWAY, distances, np.inf), axis=1,
            initial=np.inf,
        )

        valid = np.isfinite(home_distance) & np.isfinite(away_distance)
        margin = np.zeros(len(ball), dtype=np.float64)
        margin[valid] = away_distance[valid] - home_distance[valid]
        home_probability = 1.0 / (1.0 + np.exp(-np.clip(
            margin / self.temperature_m, -40.0, 40.0
        )))
        # Distance-only cues cannot establish a stoppage. Missing-ball and
        # missing-player rows therefore remain neutral H/A evidence.
        home_probability[~valid] = 0.5
        output = np.column_stack((home_probability, 1.0 - home_probability,
                                  np.zeros(len(ball), dtype=np.float64)))
        if "dead" in track:
            dead = np.asarray(track["dead"])
            if dead.shape != (len(ball),):
                raise ValueError("track['dead'] must have shape (n_frames,)")
            if dead.dtype.kind == "b":
                dead = dead.astype(bool, copy=False)
            elif (dead.dtype.kind in "iuf" and np.isfinite(dead).all()
                  and np.isin(dead, (0, 1)).all()):
                dead = dead.astype(bool)
            else:
                raise ValueError("track['dead'] must contain only booleans or 0/1")
            output[dead] = (0.0, 0.0, 1.0)
        return normalize(output)


class NearestPlayerLong(NearestPlayer):
    """Longer ball-gap carry variant, retained for baseline comparisons."""

    name = "nearest_player_long"

    def __init__(self):
        super().__init__(ffill_s=6.0, loose_m=9.0)
