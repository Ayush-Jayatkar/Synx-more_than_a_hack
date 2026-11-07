"""Module A. Load tracking into per-frame arrays, in metres, indexed by video frame."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (official slot -> team mapping)

HOME, AWAY, NOBODY = 0, 1, -1


def parse_atd(raw_path, n_slots, n_frames, pitch):
    """(ball, players) from the raw ATD. ball (n, 2), players (n, n_slots, 2), metres, NaN = missing.

    Line format (see starter/baseline.py): "<frame_id>:<x,y,..>;<x,y,..>;...:<x,y,..>", 0-1 coords.
    """
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

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
            players[frame] = values.reshape(n_slots, -1)[:, :2] * scale
            ball[frame] = np.array(parts[2].split(",")[:2], dtype=np.float32) * scale
    return ball, players


def load(game=GAME, data_dir=None):
    """dict with ball (n,2), players (n,slots,2), team (slots,), pitch (length, width)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    pitch = GAMES[game]["pitch"]
    ball, players = parse_atd(raw_path, len(team), source_frames("tactical", game), pitch)
    return {"ball": ball, "players": players, "team": team, "pitch": pitch}
