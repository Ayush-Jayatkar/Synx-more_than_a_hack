"""Stoppage (dead-ball) evidence beyond the basic half-window and ball-out rules.

Two sources, both fixed in advance from football reasoning and never fitted to ground truth:

* Tracking (backup, needs nothing but the ATD):
    - still ball: the ball is seen and at rest for STILL_MIN_FRAMES. In open play the ball is
      almost never motionless for 2 s; a ball placed for a free kick, goal kick or corner is.
    - out-of-play bridge: once the ball is seen beyond a line, play stays stopped while the ball
      is unseen (being retrieved off camera), up to MAX_OUT_BRIDGE_FRAMES.
* Smart Tagging set pieces (official asset, timed to the tactical video):
    every set piece is a restart after a stoppage. The tag window is a fixed 35 s (5 s for kick
    offs), so it is never copied across as D. Tracking locates the stoppage inside it: from the
    first stoppage evidence near the window start until the ball is struck back into play.

All inputs are whole-match arrays, so the mask is computed once before chunked inference and the
labels stay independent of chunk size.
"""
import numpy as np

from synx import data, pitch
from synx.possession import default_dead_mask

FPS = data.FPS
STILL_SPEED_MS = 0.5            # below this the ball is at rest, allowing for tracking jitter
STILL_MIN_FRAMES = 2 * FPS      # 2 s at rest: a placed ball, not a ball in open play
TAGGED_STILL_MIN_FRAMES = FPS   # 1 s is enough when a set-piece tag confirms a restart nearby
MAX_OUT_BRIDGE_FRAMES = 30 * FPS  # a restart normally follows the ball leaving the pitch within 30 s
RESTART_SPEED_MS = 5.0          # a throw or kick moves the ball faster than a player carrying it
RESTART_MIN_FRAMES = 10         # 0.4 s of that speed, so a single noisy frame is not a restart
TAG_LOOKBACK_FRAMES = 10 * FPS  # the stoppage can begin shortly before the tagged window
TAG_EVIDENCE_FRAMES = 15 * FPS  # ... and must begin within the first 15 s of it
GOAL_LOOKBACK_FRAMES = 120 * FPS  # a kick off after a goal follows the goal within 2 minutes
SET_PIECE_CODE = "set pieces"
KICKOFF_AFTER_GOAL = "kick off (after goal)"
KICKOFF_START = "kick off (start)"
MODES = ("basic", "tracking", "tagged")


def _runs(mask):
    """[(start, end)] half-open runs of True."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _long_runs(mask, min_frames):
    out = np.zeros(len(mask), dtype=bool)
    for start, end in _runs(mask):
        if end - start >= min_frames:
            out[start:end] = True
    return out


def _signals(track):
    ball = np.asarray(track["ball"], dtype=float)
    seen = np.isfinite(ball).all(axis=1)
    if "pitch" not in track:
        raise ValueError("tracking stoppage evidence needs track['pitch']")
    speed = np.asarray(track["ball_speed_filtered"] if "ball_speed_filtered" in track
                       else data.filter_ball_speed(track["ball_speed"]), dtype=float)
    with np.errstate(invalid="ignore"):
        at_rest = seen & (speed < STILL_SPEED_MS)
        fast = seen & (speed >= RESTART_SPEED_MS)
    out = pitch.ball_out_of_pitch(ball, track["pitch"])
    return seen, at_rest, fast & ~out, out


def still_ball(at_rest, min_frames=STILL_MIN_FRAMES):
    """Frames inside a run of at-rest ball frames at least ``min_frames`` long."""
    return _long_runs(at_rest, min_frames)


def out_bridge(out, seen, max_frames=MAX_OUT_BRIDGE_FRAMES):
    """Unseen frames that directly follow a seen-out frame, capped at ``max_frames``."""
    bridge = np.zeros(len(out), dtype=bool)
    for start, end in _runs(~seen):
        if start > 0 and out[start - 1]:
            bridge[start:min(end, start + max_frames)] = True
    return bridge


def restart_frame(fast_inside, start, end):
    """First frame in [start, end) that begins RESTART_MIN_FRAMES of fast ball inside the pitch."""
    for run_start, run_end in _runs(fast_inside[start:end]):
        if run_end - run_start >= RESTART_MIN_FRAMES:
            return start + run_start
    return None


def tagged_stoppages(events, out, at_rest, fast_inside, n_frames):
    """D intervals located by tracking inside Smart Tagging set-piece windows.

    Returns (mask, counts). Events whose window shows no stoppage evidence are skipped rather
    than guessed. Kick offs at the start of a half are covered by the in-play window already.
    """
    mask = np.zeros(n_frames, dtype=bool)
    evidence = out | _long_runs(at_rest, TAGGED_STILL_MIN_FRAMES)
    counts = {"set_pieces": 0, "used": 0, "restart_found": 0, "after_goal_used": 0, "skipped": 0}
    for event in events:
        try:
            if str(event.get("code", "")).strip().lower() != SET_PIECE_CODE:
                continue
            tags = event.get("tags") or {}
            kind = str(tags.get("Type", "")).strip().lower()
            start, end = int(event["start_frame"]), int(event["end_frame"])
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            counts["skipped"] += 1
            continue
        if kind == KICKOFF_START:
            continue
        counts["set_pieces"] += 1
        start, end = max(start, 0), min(end, n_frames)
        if end <= start:
            counts["skipped"] += 1
            continue
        if kind == KICKOFF_AFTER_GOAL:
            # After a goal the ball is dead until the kick off: from the last time the ball was
            # seen beyond a line (the goal) to the restart inside the short kick-off window.
            lo = max(start - GOAL_LOOKBACK_FRAMES, 0)
            hits = np.flatnonzero(out[lo:start])
            if not len(hits):
                counts["skipped"] += 1
                continue
            first = lo + hits[-1]
            restart = restart_frame(fast_inside, start, end)
            mask[first:restart if restart is not None else start] = True
            counts["after_goal_used"] += 1
            counts["restart_found"] += restart is not None
            continue
        lo = max(start - TAG_LOOKBACK_FRAMES, 0)
        hits = np.flatnonzero(evidence[lo:min(start + TAG_EVIDENCE_FRAMES, end)])
        if not len(hits):
            counts["skipped"] += 1
            continue
        first = lo + hits[0]
        restart = restart_frame(fast_inside, first, end)
        if restart is None:
            # No clear restart in the window: keep only the evidence frames themselves.
            mask[first:end] |= evidence[first:end]
        else:
            mask[first:restart] = True
            counts["restart_found"] += 1
        counts["used"] += 1
    return mask, counts


def dead_mask(track, mode="basic", events=None):
    """(dead, report) for ``mode`` in MODES. ``basic`` is exactly ``default_dead_mask``."""
    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    dead = default_dead_mask(track)
    report = {"mode": mode, "basic": int(dead.sum())}
    if mode == "basic":
        return dead, report
    seen, at_rest, fast_inside, out = _signals(track)
    extra = still_ball(at_rest) | out_bridge(out, seen)
    report["tracking_added"] = int((extra & ~dead).sum())
    dead = dead | extra
    if mode == "tagged":
        if events is None:
            raise ValueError("mode 'tagged' needs Smart Tagging events")
        tagged, counts = tagged_stoppages(events, out, at_rest, fast_inside, len(dead))
        report["tagged_added"] = int((tagged & ~dead).sum())
        report.update(counts)
        dead = dead | tagged
    report["total"] = int(dead.sum())
    return dead, report
