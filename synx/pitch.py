"""Module B. Where the ball is relative to the pitch, and when the match is being played.

The tactical homography is valid on every frame and the ATD is already in pitch coordinates,
so v1 does not need it. Smart Tagging is not used in v1 (see Synx.md).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402  (published kickoff frames)

# Ball must be clearly over the line: ball radius ~0.11 m plus tracking noise.
OUT_MARGIN_M = 0.5


def in_play_window(n_frames, game=GAME, start_frame=0):
    """Return in-play flags for ``[start_frame, start_frame+n_frames)``.

    ``start_frame`` is the absolute tactical-video frame index. Keeping this
    offset explicit makes the same mask usable for a whole match or a chunk.
    """
    if (not isinstance(n_frames, (int, np.integer)) or n_frames < 0
            or not isinstance(start_frame, (int, np.integer)) or start_frame < 0):
        raise ValueError("n_frames and start_frame must be nonnegative integers")
    mask = np.zeros(n_frames, dtype=bool)
    end_frame = start_frame + n_frames
    kickoffs = sorted(KICKOFF_VIDEO_FRAMES.values())
    for kickoff, (_, _, last) in zip(kickoffs, half_video_blocks(game)):
        first = max(kickoff, start_frame)
        last_exclusive = min(last + 1, end_frame)
        if first < last_exclusive:
            mask[first - start_frame:last_exclusive - start_frame] = True
    return mask


def ball_out_of_pitch(ball, pitch, margin=OUT_MARGIN_M):
    """True where the ball is seen beyond a touchline or goal line. False where it is missing."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
