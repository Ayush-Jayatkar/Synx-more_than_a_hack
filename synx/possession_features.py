"""Vectorised team features for possession estimation; slots are transient tracklets."""
import numpy as np
from synx.data import HOME, AWAY, FPS
DEFAULT_CONTROL_RADIUS_M = 2.0


def _team_features(ball, players, speeds, team, team_id, radius):
    n = len(ball)
    selected = players[:, team == team_id, :]
    selected_speed = speeds[:, team == team_id]
    if selected.shape[1]:
        delta = selected - ball[:, None, :]
        distances = np.hypot(delta[..., 0], delta[..., 1])
        distances[~np.isfinite(delta).all(axis=-1)] = np.inf
        ordered = np.sort(distances, axis=1)
        nearest = ordered[:, 0].copy()
        second = ordered[:, 1].copy() if ordered.shape[1] > 1 else np.full(n, np.inf)
        idx = np.argmin(distances, axis=1)
        nearest_speed = selected_speed[np.arange(n), idx].copy()
        nearest_speed[~np.isfinite(nearest)] = np.nan
        nearby = np.sum(distances <= radius, axis=1).astype(np.int32)
    else:
        nearest = second = nearest_speed = np.full(n, np.nan)
        nearby = np.zeros(n, dtype=np.int32)
    nearest[~np.isfinite(nearest)] = np.nan
    second[~np.isfinite(second)] = np.nan
    return nearest, second, nearest_speed, nearby


def build_features(track, control_radius=DEFAULT_CONTROL_RADIUS_M,
                   previous=None, return_state=False):
    """Build arrays aligned to tracking frames. Distances are m, speeds m/s.

    Inputs are data.load() arrays: ball (F,2), players (F,S,2), player_speed
    (F,S), team (S,), ball_speed (F,). Returns one value per frame, vectors
    with shape (F,2). Positive distance_advantage means HOME is closer.
    """
    required = ("ball", "players", "player_speed", "team", "ball_speed")
    missing = [k for k in required if k not in track]
    if missing:
        raise ValueError("Missing tracking fields: {}".format(missing))
    ball = np.asarray(track["ball"], dtype=float)
    players = np.asarray(track["players"], dtype=float)
    speeds = np.asarray(track["player_speed"], dtype=float)
    team = np.asarray(track["team"])
    ball_speed = np.asarray(track["ball_speed"], dtype=float)
    if ball.ndim != 2 or ball.shape[1:] != (2,):
        raise ValueError("ball must have shape (frames, 2)")
    if players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2:
        raise ValueError("players must have shape (frames, slots, 2) aligned to ball")
    if speeds.shape != players.shape[:2]:
        raise ValueError("player_speed must have shape (frames, slots)")
    if team.ndim != 1 or len(team) != players.shape[1] or np.any(~np.isin(team, (HOME, AWAY))):
        raise ValueError("team must contain HOME/AWAY for every slot")
    if ball_speed.shape != (len(ball),):
        raise ValueError("ball_speed must have shape (frames,)")
    if not np.isfinite(control_radius) or control_radius < 0:
        raise ValueError("control_radius must be finite and nonnegative")
    ball = ball.copy()
    ball[~np.isfinite(ball).all(axis=1)] = np.nan
    speeds = speeds.copy()
    speeds[~np.isfinite(speeds) | (speeds < 0)] = np.nan
    h = _team_features(ball, players, speeds, team, HOME, control_radius)
    a = _team_features(ball, players, speeds, team, AWAY, control_radius)
    hn, hs, hv, hc = h
    an, ass, av, ac = a
    advantage = an - hn
    advantage[~(np.isfinite(an) & np.isfinite(hn))] = np.nan
    n = len(ball)
    velocity = np.full((n, 2), np.nan)
    if n and previous is not None:
        previous_ball = np.asarray(previous.get("ball", [np.nan, np.nan]), dtype=float)
        if previous_ball.shape == (2,) and np.isfinite(previous_ball).all() and np.isfinite(ball[0]).all():
            velocity[0] = (ball[0] - previous_ball) * FPS
    if n > 1:
        delta = ball[1:] - ball[:-1]
        valid = np.isfinite(delta).all(axis=1)
        velocity[1:][valid] = delta[valid] * FPS
    direction = np.full_like(velocity, np.nan)
    magnitude = np.hypot(velocity[:, 0], velocity[:, 1])
    moving = np.isfinite(magnitude) & (magnitude > 0)
    direction[moving] = velocity[moving] / magnitude[moving, None]
    def approach(distance, previous_value):
        result = np.full(n, np.nan)
        if n and previous_value is not None and np.isfinite(previous_value) and np.isfinite(distance[0]):
            result[0] = (previous_value - distance[0]) * FPS
        valid = np.isfinite(distance[1:]) & np.isfinite(distance[:-1])
        result[1:][valid] = (distance[:-1][valid] - distance[1:][valid]) * FPS
        return result
    result = {
        "home_nearest_distance": hn.astype(np.float32), "away_nearest_distance": an.astype(np.float32),
        "home_second_distance": hs.astype(np.float32), "away_second_distance": ass.astype(np.float32),
        "home_nearest_speed": hv.astype(np.float32), "away_nearest_speed": av.astype(np.float32),
        "home_players_nearby": hc, "away_players_nearby": ac,
        "distance_advantage": advantage.astype(np.float32),
        "ball_speed": np.where(np.isfinite(ball_speed) & (ball_speed >= 0), ball_speed, np.nan).astype(np.float32),
        "ball_velocity": velocity.astype(np.float32), "ball_direction": direction.astype(np.float32),
        "home_approach_speed": approach(
            hn, None if previous is None else previous.get("home_nearest_distance")
        ).astype(np.float32),
        "away_approach_speed": approach(
            an, None if previous is None else previous.get("away_nearest_distance")
        ).astype(np.float32),
        "ball_seen": np.isfinite(ball).all(axis=1),
    }
    if not return_state:
        return result
    state = previous
    if n:
        state = {
            "ball": ball[-1].copy(),
            "home_nearest_distance": float(hn[-1]),
            "away_nearest_distance": float(an[-1]),
        }
    return result, state
