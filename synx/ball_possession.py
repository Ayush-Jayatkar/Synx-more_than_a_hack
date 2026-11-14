"""Team-level ball proximity and configurable control heuristics."""
import numpy as np
from synx.data import AWAY, HOME, NOBODY
CONTROL_RADIUS_M = 2.0


def team_distances(ball, players, team):
    """Return nearest HOME/AWAY distance (m), using inf when unavailable.

    Shapes: ball (F,2), players (F,S,2), team (S,). Slots are not identities.
    """
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    team = np.asarray(team)
    if ball.ndim != 2 or ball.shape[1:] != (2,):
        raise ValueError("ball must have shape (frames, 2)")
    if players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2:
        raise ValueError("players must have shape (frames, slots, 2)")
    if team.shape != (players.shape[1],):
        raise ValueError("team must have shape (slots,)")
    delta = players - ball[:, None, :]
    d = np.hypot(delta[..., 0], delta[..., 1])
    d[~np.isfinite(delta).all(axis=-1)] = np.inf
    result = []
    for side in (HOME, AWAY):
        side_d = d[:, team == side]
        result.append(side_d.min(axis=1) if side_d.shape[1] else np.full(len(ball), np.inf))
    return tuple(result)


def control(d_home, d_away, radius=CONTROL_RADIUS_M, contest_margin=0.4):
    """Threshold baseline. Radius is m; margin is required separation in m.

    NOBODY represents uncertain/uncontrolled live-ball evidence, never dead ball.
    """
    home, away = np.asarray(d_home, dtype=float), np.asarray(d_away, dtype=float)
    if home.shape != away.shape or home.ndim != 1:
        raise ValueError("team distances must be aligned one-dimensional arrays")
    if not np.isfinite(radius) or radius < 0 or not np.isfinite(contest_margin) or contest_margin < 0:
        raise ValueError("radius and contest_margin must be finite and nonnegative")
    out = np.full(home.shape, NOBODY, dtype=np.int8)
    out[(home <= radius) & (away - home > contest_margin)] = HOME
    out[(away <= radius) & (home - away > contest_margin)] = AWAY
    return out


def enhanced_control(features, radius=CONTROL_RADIUS_M, contest_margin=0.4,
                     competition_penalty_m=0.25, approach_weight=0.05,
                     speed_uncertainty_m_per_mps=0.02, max_uncertainty_m=1.0):
    """Estimate control from per-frame tactical features, returning codes.

    A team must have a nearby player. The score is nearest distance plus a
    per-extra-nearby-player competition penalty, minus a small bounded credit
    for aggregate approach evidence. Fast ball movement expands the contest
    margin (uncertainty), capped in metres. All parameters are physical
    heuristic scales, not tuned or claimed optimal.
    """
    params = (radius, contest_margin, competition_penalty_m, approach_weight,
              speed_uncertainty_m_per_mps, max_uncertainty_m)
    if any(not np.isfinite(x) or x < 0 for x in params):
        raise ValueError("all thresholds must be finite and nonnegative")
    def arr(name):
        if name not in features:
            raise ValueError("Missing feature: " + name)
        return np.asarray(features[name], dtype=float)
    hn, an = arr("home_nearest_distance"), arr("away_nearest_distance")
    hc, ac = arr("home_players_nearby"), arr("away_players_nearby")
    hs, ass = arr("home_approach_speed"), arr("away_approach_speed")
    bs = arr("ball_speed")
    if any(x.ndim != 1 or x.shape != hn.shape for x in (an, hc, ac, hs, ass, bs)):
        raise ValueError("control features must be aligned one-dimensional arrays")
    hscore = hn + competition_penalty_m * np.maximum(hc - 1, 0) - approach_weight * np.clip(np.nan_to_num(hs, nan=0.0, posinf=0.0, neginf=0.0), 0, 5)
    ascore = an + competition_penalty_m * np.maximum(ac - 1, 0) - approach_weight * np.clip(np.nan_to_num(ass, nan=0.0, posinf=0.0, neginf=0.0), 0, 5)
    uncertainty = np.minimum(np.nan_to_num(bs, nan=0.0, posinf=0.0), 50.0) * speed_uncertainty_m_per_mps
    margin = contest_margin + np.minimum(uncertainty, max_uncertainty_m)
    out = np.full(hn.shape, NOBODY, dtype=np.int8)
    out[(hn <= radius) & (hscore + margin < ascore)] = HOME
    out[(an <= radius) & (ascore + margin < hscore)] = AWAY
    return out
