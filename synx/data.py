"""Module A (Workstream 1). Load tracking into per-frame arrays, in metres, indexed by video frame.

    track = load()                       # parses once, then reads cache/BAR-ATH_tracking.npz
    track = load(ball_speed_window=5)    # also adds ball_speed_filtered
    python -m synx.data                  # data check: aggregate stats to read locally
    python -m synx.data --schema         # ball channels declared by the official ATD metadata

load() returns a dict. These keys are always present and unchanged for Workstream 2:
    ball          (n, 2)        float32  ball x, y in metres; NaN where the ball is not seen
    players       (n, slots, 2) float32  player x, y in metres; NaN where the slot is empty
    player_speed  (n, slots)    float32  speed in m/s as reported by the tracking; NaN where empty
    ball_speed    (n,)          float32  raw ball speed in m/s from frame-to-frame movement;
                                         NaN on frame 0 and wherever either frame lacks the ball
    team          (slots,)      int8     HOME (0) or AWAY (1) for each track slot
    pitch         (length, width)        metres
Optional keys, added only when available or requested:
    ball_z               (n,)   float32  ball height exactly as reported (not rescaled); only when the
                                         metadata schema declares a ball "z" channel; NaN where missing
    ball_speed_filtered  (n,)   float32  median-filtered ball speed in m/s; only with ball_speed_window

Row i is video frame i (= submission row i). Slots are tracklets, not people: they appear, vanish
and swap. Use them only by team.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (official slot -> team mapping)

HOME, AWAY, NOBODY = 0, 1, -1
FPS = 25
CACHE_DIR = os.path.join(ROOT, "cache")  # gitignored: derived from the match data, never commit
CACHE_VERSION = 2  # bump when the cached arrays change, so old caches rebuild
ARRAYS = ("ball", "players", "player_speed", "team")
FILTER_WINDOWS = (3, 5)


def ball_channels(meta_path):
    """Ball channel ids, in order, from the first data-format specification of the EPTS metadata.

    The FIFA/EPTS schema lists each ball field as <BallChannelRef channelId="..."/> (this is how
    kloppy builds its parser). Returns e.g. ["x", "y"] or ["x", "y", "z"]; [] if none are declared.
    """
    root = ET.parse(meta_path).getroot()
    spec = next(root.iter("DataFormatSpecification"), root)
    return [elm.get("channelId") for elm in spec.iter("BallChannelRef")]


def parse_atd(raw_path, n_slots, n_frames, pitch, ball_z_index=None):
    """(ball, players, player_speed) from the raw ATD, plus ball_z when ball_z_index is given.

    Line format (see starter/baseline.py): "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,..>", 0-1 coords,
    s in m/s. Shapes: ball (n, 2) and players (n, n_slots, 2) in metres, player_speed (n, n_slots),
    ball_z (n,) as reported. Lines that are not three ':'-separated parts are skipped, as in the
    official baseline. Frame ids outside 1..n_frames are ignored. Unreadable numbers or a player
    block that does not match n_slots raise ValueError naming the frame id.
    """
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_z = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
                if not 0 <= frame < n_frames:
                    continue
                values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
                ball_values = np.array(parts[2].split(","), dtype=np.float32)
            except ValueError:
                raise ValueError("frame_id {!r}: unreadable number".format(parts[0]))
            if values.size % n_slots:
                raise ValueError("frame_id {}: player groups do not match {} slots".format(
                    frame + 1, n_slots))
            groups = values.reshape(n_slots, -1)
            players[frame] = groups[:, :2] * scale
            if groups.shape[1] > 2:
                speed[frame] = groups[:, 2]
            if ball_values.size >= 2:
                ball[frame] = ball_values[:2] * scale
            if ball_z_index is not None and ball_values.size > ball_z_index:
                ball_z[frame] = ball_values[ball_z_index]
    if ball_z_index is None:
        return ball, players, speed
    return ball, players, speed, ball_z


def ball_speed(ball, fps=FPS):
    """Ball speed in m/s from consecutive frames. NaN on frame 0 and wherever either frame lacks the ball."""
    out = np.full(len(ball), np.nan, dtype=np.float32)
    if len(ball) > 1:
        out[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * fps
    return out


def filter_ball_speed(speed, window=5):
    """Centred median of `speed` (m/s) over `window` frames, never reaching across a missing value.

    Each frame uses only neighbours from its own unbroken run of finite speeds, so values on either
    side of a ball gap never mix and no speed is created inside a gap: NaN in gives NaN out. Runs
    shorter than the window use whatever neighbours they have. The input array is not modified.
    """
    if window not in FILTER_WINDOWS:
        raise ValueError("window must be one of {}".format(FILTER_WINDOWS))
    speed = np.asarray(speed, dtype=np.float32)
    valid = np.isfinite(speed)
    run_id = np.cumsum(~valid)  # constant inside each run of finite values
    half = window // 2
    padded = np.pad(speed, half, constant_values=np.nan)
    padded_run = np.pad(run_id, half, constant_values=-1)
    values = np.lib.stride_tricks.sliding_window_view(padded, window).copy()
    same_run = np.lib.stride_tricks.sliding_window_view(padded_run, window) == run_id[:, None]
    values[~same_run] = np.nan
    out = np.full(len(speed), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = np.nanmedian(values[valid], axis=1)
    return out


def _stamp(paths):
    """Cache version plus size and mtime (ns) of each source file; any change rebuilds the cache."""
    stats = [os.stat(p) for p in paths]
    return np.array([CACHE_VERSION] + [v for s in stats for v in (s.st_size, s.st_mtime_ns)],
                    dtype=np.int64)


def _read_cache(cache, stamp):
    """Cached arrays if the cache matches `stamp`, else None. A corrupt or partial cache counts as a miss."""
    try:
        with np.load(cache) as saved:
            if not np.array_equal(saved["stamp"], stamp):
                return None
            track = {name: saved[name] for name in ARRAYS}
            if "ball_z" in saved.files:
                track["ball_z"] = saved["ball_z"]
            return track
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _write_cache(cache, stamp, track):
    """Write via a temporary file and rename, so an interrupted write never leaves a broken cache."""
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, stamp=stamp, **track)
    os.replace(tmp, cache)


def load(game=GAME, data_dir=None, cache_dir=CACHE_DIR, use_cache=True, ball_speed_window=None):
    """Tracking dict (see module docstring). Parses once, then reuses cache_dir/<game>_tracking.npz."""
    paths = require(["atd_meta", "atd_raw"], game, data_dir)
    stamp = _stamp(paths)
    cache = os.path.join(cache_dir, "{}_tracking.npz".format(game))

    track = _read_cache(cache, stamp) if use_cache and os.path.isfile(cache) else None
    if track is None:
        meta_path, raw_path = paths
        team = team_of_each_track(meta_path)
        channels = ball_channels(meta_path)
        z_index = channels.index("z") if "z" in channels else None
        parsed = parse_atd(raw_path, len(team), source_frames("tactical", game),
                           GAMES[game]["pitch"], ball_z_index=z_index)
        track = {"ball": parsed[0], "players": parsed[1], "player_speed": parsed[2], "team": team}
        if z_index is not None:
            track["ball_z"] = parsed[3]
        if use_cache:
            _write_cache(cache, stamp, track)

    track["ball_speed"] = ball_speed(track["ball"])
    if ball_speed_window is not None:
        track["ball_speed_filtered"] = filter_ball_speed(track["ball_speed"], ball_speed_window)
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
    empty = len(ball) == 0
    return {
        "frames": len(ball),
        "ball_seen_pct": 0.0 if empty else round(100.0 * seen.mean(), 1),
        "longest_ball_gap_s": round(gaps.max() / float(FPS), 1) if gaps.size else 0.0,
        "mean_tracked_players": 0.0 if empty else round(float(tracked.mean()), 1),
        "frames_over_22_tracked_pct": 0.0 if empty else round(100.0 * (tracked > 22).mean(), 1),
        "home_slots": int((track["team"] == HOME).sum()),
        "away_slots": int((track["team"] == AWAY).sum()),
        "ball_height_available": "ball_z" in track,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workstream 1 local data checks.")
    parser.add_argument("--schema", action="store_true",
                        help="print the ball channels the ATD metadata declares, then exit")
    args = parser.parse_args()
    if args.schema:
        meta_path = require(["atd_meta"])[0]
        channels = ball_channels(meta_path)
        print("ball channels declared by the metadata: {}".format(channels or "none"))
        print("ball height (z) available: {}".format("z" in channels))
        raise SystemExit(0)
    from games import half_video_blocks
    track = load()
    graded = np.concatenate([np.arange(a, b + 1) for _, a, b in half_video_blocks(GAME)])
    for name, value in check(track, graded).items():
        print("{:<28} {}".format(name, value))
