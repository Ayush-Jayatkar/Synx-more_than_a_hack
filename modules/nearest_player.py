"""Baseline module: the team of the player nearest the ball has possession.
Ball loose or unseen (> ffill_s seconds) -> dead ball. Deliberately simple: it is the
reference score every other module has to beat."""
import warnings
import numpy as np
from interfaces import PossessionModule, HOME, AWAY, normalize


def _ffill(ball, max_gap):
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    ok = (last >= 0) & (np.arange(n) - last <= max_gap)
    out = np.full_like(ball, np.nan)
    out[ok] = ball[last[ok]]
    return out


class NearestPlayer(PossessionModule):
    name = "nearest_player"

    def __init__(self, ffill_s=2.0, fps=25, loose_m=6.0):
        self.max_gap = int(ffill_s * fps)
        self.loose_m = loose_m

    def predict_proba(self, track):
        ball = _ffill(np.asarray(track["ball"], dtype=np.float64), self.max_gap)
        pl = np.asarray(track["players"], dtype=np.float64)
        team = np.asarray(track["team"])
        d = np.hypot(pl[..., 0] - ball[:, None, 0], pl[..., 1] - ball[:, None, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dh = np.nanmin(np.where(team[None, :] == HOME, d, np.nan), axis=1)
            da = np.nanmin(np.where(team[None, :] == AWAY, d, np.nan), axis=1)
        valid = ~np.isnan(dh) & ~np.isnan(da)
        margin = np.where(valid, da - dh, 0.0)              # > 0: home is closer
        dmin = np.where(valid, np.minimum(dh, da), 99.0)
        p_home = 1 / (1 + np.exp(-margin / 2.0))
        p_dead = 1 / (1 + np.exp(-(dmin - self.loose_m) / 1.5))
        return normalize(np.stack([(1 - p_dead) * p_home,
                                   (1 - p_dead) * (1 - p_home),
                                   p_dead], axis=1))

class NearestPlayerLong(NearestPlayer):
    name = "nearest_player_long"
    def __init__(self):
        super().__init__(ffill_s=6.0, loose_m=9.0)