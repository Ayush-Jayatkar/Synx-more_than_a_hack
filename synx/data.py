"""Module A (Workstream 1). Load tracking into per-frame arrays, in metres, indexed by video frame.

    track = load()             # parses once, then reads cache/BAR-ATH_tracking.npz
    python -m synx.data        # data check: prints aggregate stats for you to read locally

load() returns a dict:
    ball          (n, 2)        ball x, y in metres; NaN where the ball is not seen
    players       (n, slots, 2) player x, y in metres; NaN where the slot is empty
    player_speed  (n, slots)    speed in m/s as reported by the tracking; NaN where empty
    ball_speed    (n,)          ball speed in m/s from frame-to-frame movement; NaN if either frame lacks the ball
    team          (slots,)      HOME (0) or AWAY (1) for each track slot
    pitch         (length, width) in metres

Row i is video frame i (= submission row i). Slots are tracklets, not people: they appear, vanish
and swap. Use them only by team.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (official slot -> team mapping)

HOME, AWAY, NOBODY = 0, 1, -1
FPS = 25
CACHE_DIR = os.path.join(ROOT, "cache")  # gitignored: derived from the match data, never commit
ARRAYS = ("ball", "players", "player_speed", "team")


def parse_atd(raw_path, n_slots, n_frames, pitch):
    """(ball, players, player_speed) from the raw ATD.

    Line format (see starter/baseline.py): "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,..>", 0-1 coords,
    s in m/s. Shapes: ball (n, 2) and players (n, n_slots, 2) in metres, player_speed (n, n_slots).
    """
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if values.size % n_slots:
                raise ValueError("frame_id {}: player groups do not match {} slots".format(
                    frame + 1, n_slots))
            groups = values.reshape(n_slots, -1)
            players[frame] = groups[:, :2] * scale
            if groups.shape[1] > 2:
                speed[frame] = groups[:, 2]
            ball[frame] = np.array(parts[2].split(",")[:2], dtype=np.float32) * scale
    return ball, players, speed


def ball_speed(ball, fps=FPS):
    """Ball speed in m/s from consecutive frames. NaN on frame 0 and wherever either frame lacks the ball."""
    out = np.full(len(ball), np.nan, dtype=np.float32)
    out[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * fps
    return out


def _stamp(paths):
    """Size and mtime of the source files, so a changed file invalidates the cache."""
    return np.array([v for p in paths for v in (os.path.getsize(p), os.path.getmtime(p))])


def load(game=GAME, data_dir=None, cache_dir=CACHE_DIR, use_cache=True):
    """Tracking dict (see module docstring). Parses once, then reuses cache_dir/<game>_tracking.npz."""
    paths = require(["atd_meta", "atd_raw"], game, data_dir)
    stamp = _stamp(paths)
    cache = os.path.join(cache_dir, "{}_tracking.npz".format(game))

    track = None
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as saved:
            if np.array_equal(saved["stamp"], stamp):
                track = {name: saved[name] for name in ARRAYS}
    if track is None:
        meta_path, raw_path = paths
        team = team_of_each_track(meta_path)
        ball, players, speed = parse_atd(raw_path, len(team), source_frames("tactical", game),
                                         GAMES[game]["pitch"])
        track = {"ball": ball, "players": players, "player_speed": speed, "team": team}
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, stamp=stamp, **track)

    track["ball_speed"] = ball_speed(track["ball"])
    track["pitch"] = GAMES[game]["pitch"]
    return track


def check(track, frames=None):
    """Aggregate data-quality stats over `frames` (default all). For reading locally, never sharing."""
    ball = track["ball"] if frames is None else track["ball"][frames]
    players = track["players"] if frames is None else track["players"][frames]
    seen = ~np.isnan(ball[:, 0])
    tracked = (~np.isnan(players[:, :, 0])).sum(axis=1)
    edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    gaps = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {
        "frames": len(ball),
        "ball_seen_pct": round(100.0 * seen.mean(), 1),
        "longest_ball_gap_s": round(gaps.max() / float(FPS), 1) if gaps.size else 0.0,
        "mean_tracked_players": round(float(tracked.mean()), 1),
        "frames_over_22_tracked_pct": round(100.0 * (tracked > 22).mean(), 1),
        "home_slots": int((track["team"] == HOME).sum()),
        "away_slots": int((track["team"] == AWAY).sum()),
    }


if __name__ == "__main__":
    from games import half_video_blocks
    track = load()
    graded = np.concatenate([np.arange(a, b + 1) for _, a, b in half_video_blocks(GAME)])
    for name, value in check(track, graded).items():
        print("{:<28} {}".format(name, value))
