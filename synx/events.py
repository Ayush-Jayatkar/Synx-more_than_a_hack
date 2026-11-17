"""Optional Smart Tagging interval alignment; no possession labels are read.

The parser adapts the repository's permitted Smart Tagging loader. Events are
context only: phases and set-piece tags are not treated as possession ground
truth and do not automatically create dead-ball labels.
"""
import numpy as np


def align_events(events, n_frames, team_to_side=None):
    """Convert loader-style events to frame arrays with defensive validation.

    Each event may contain ``start_frame``, ``end_frame``, ``code`` and a
    ``tags`` mapping. Intervals are half-open. Unknown or malformed events are
    skipped and counted; unknown teams remain -1.
    """
    if not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
        raise ValueError("n_frames must be a nonnegative integer")
    team_to_side = dict(team_to_side or {})
    context = {
        "set_piece": np.zeros(n_frames, dtype=bool),
        "event_team": np.full(n_frames, -1, dtype=np.int8),
        "event_present": np.zeros(n_frames, dtype=bool),
    }
    skipped = 0
    for event in events:
        try:
            start = int(event["start_frame"])
            end = int(event["end_frame"])
            if start < 0 or end < start:
                raise ValueError
            tags = event.get("tags") or {}
            lo, hi = min(start, n_frames), min(end, n_frames)
            if lo >= hi:
                continue
            context["event_present"][lo:hi] = True
            if str(event.get("code", "")).strip().lower() == "set pieces":
                context["set_piece"][lo:hi] = True
            team = tags.get("Team")
            side = team_to_side.get(team, -1)
            context["event_team"][lo:hi] = side if side in (0, 1) else -1
        except (KeyError, TypeError, ValueError, OverflowError):
            skipped += 1
    context["skipped_events"] = skipped
    return context


def load_smart_tagging_events(game=None, data_dir=None):
    """Load optional event intervals through the official starter parser.

    No XML is printed or copied. The returned intervals remain local to the
    prediction process and can be passed through ``align_events``.
    """
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    starter = os.path.join(root, "starter")
    if starter not in sys.path:
        sys.path.insert(0, starter)
    import load_smart_tagging
    return load_smart_tagging.load(game=game or load_smart_tagging.GAME,
                                   data_dir=data_dir)
