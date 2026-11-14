"""Local, label-free engineering diagnostics for prediction behaviour."""
import csv
import json
import os
import time

import numpy as np

from synx import approaches, possession_features
from synx.data import AWAY, HOME, NOBODY
from synx.streaming import StreamingPossessionEngine


def _transition_count(labels):
    labels = np.asarray(labels)
    return int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0


def _spell_durations(labels, fps=25):
    labels = np.asarray(labels)
    spans = []
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        if labels[start] in ("H", "A"):
            spans.append((end - start) / float(fps))
        start = end
    if not spans:
        return {"count": 0, "mean_s": None, "median_s": None,
                "min_s": None, "max_s": None}
    values = np.asarray(spans)
    return {"count": len(spans), "mean_s": float(values.mean()),
            "median_s": float(np.median(values)), "min_s": float(values.min()),
            "max_s": float(values.max())}


def summarize_labels(labels, fps=25):
    """Summarize a prediction's structure without comparing to ground truth."""
    labels = np.asarray(labels)
    return {
        "frames": int(len(labels)),
        "label_counts": {label: int(np.sum(labels == label))
                         for label in ("H", "A", "D", "X")},
        "transitions": _transition_count(labels),
        "possession_spell_durations": _spell_durations(labels, fps),
    }


def _peak_rss_mb():
    """Return peak process working set/RSS where the platform exposes it."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return counters.PeakWorkingSetSize / (1024.0 ** 2) if ok else None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 ** 2) if os.uname().sysname == "Darwin" else peak / 1024.0
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def compare_approaches(track, dead, fps=25, switch_frames=13):
    """Compare approach behaviour, disagreement, confidence and missing handling.

    The output includes no ground-truth metric and must not be read as an
    accuracy ranking.
    """
    features = possession_features.build_features(track)
    ball_seen = features["ball_seen"]
    outputs = approaches.compare_controls(features, dead, ball_seen, switch_frames)
    names = sorted(outputs)
    report = {"frames": int(len(ball_seen)), "approaches": {}, "pairwise": {}}
    for name in names:
        values = outputs[name]
        labels = values["labels"]
        conf = values["confidence"]
        finite_conf = conf[np.isfinite(conf)]
        report["approaches"][name] = {
            **summarize_labels(labels, fps),
            "instant_control_counts": {
                "H": int(np.sum(values["control"] == HOME)),
                "A": int(np.sum(values["control"] == AWAY)),
                "NOBODY": int(np.sum(values["control"] == NOBODY)),
            },
            "confidence_score_m": {
                "finite_frames": int(len(finite_conf)),
                "mean": float(finite_conf.mean()) if len(finite_conf) else None,
                "p10": float(np.quantile(finite_conf, 0.10)) if len(finite_conf) else None,
                "median": float(np.median(finite_conf)) if len(finite_conf) else None,
                "p90": float(np.quantile(finite_conf, 0.90)) if len(finite_conf) else None,
            },
            "frame_control_runtime_s": float(values["runtime_s"]),
            "temporal_runtime_s": float(values["temporal_runtime_s"]),
            "unseen_ball_frames": int(np.sum(~ball_seen)),
            "unseen_frames_labeled_X": int(np.sum((~ball_seen) & (labels == "X"))),
        }
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            report["pairwise"][left + "__vs__" + right] = {
                "disagreement_rate": float(np.mean(outputs[left]["labels"] != outputs[right]["labels"]))
                if len(ball_seen) else 0.0,
                "left_only_transitions": _transition_count(outputs[left]["labels"]),
                "right_only_transitions": _transition_count(outputs[right]["labels"]),
            }
    return report, outputs, features


def benchmark_chunked(track, dead, chunk_size=4096, fps=25,
                      feature_kwargs=None, control_kwargs=None,
                      switch_frames=13):
    """Measure chunked combined inference and report 25-FPS throughput capacity."""
    n = len(track["ball"])
    dead = np.asarray(dead, dtype=bool)
    if dead.shape != (n,):
        raise ValueError("dead must have one boolean per frame")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    engine = StreamingPossessionEngine(feature_kwargs, control_kwargs, switch_frames)
    labels = np.empty(n, dtype="<U1")
    written = np.zeros(n, dtype=bool)
    chunk_times_ms = []
    keys = ("ball", "players", "player_speed", "team", "ball_speed")
    start = time.perf_counter()
    for chunk_start in range(0, n, chunk_size):
        end = min(n, chunk_start + chunk_size)
        chunk = {key: np.asarray(track[key])[chunk_start:end] if key != "team"
                 else np.asarray(track[key]) for key in keys}
        tick = time.perf_counter()
        result = engine.feed(chunk, dead[chunk_start:end])
        chunk_times_ms.append((time.perf_counter() - tick) * 1000.0)
        lo = result.label_start
        hi = lo + len(result.labels)
        labels[lo:hi] = result.labels
        written[lo:hi] = True
    tail_start, tail = engine.finish()
    labels[tail_start:tail_start + len(tail)] = tail
    written[tail_start:tail_start + len(tail)] = True
    elapsed = time.perf_counter() - start
    if not written.all():
        raise RuntimeError("benchmark stream did not finalize every frame")
    frames = n
    rate = frames / elapsed if elapsed else float("inf")
    chunk_times = np.asarray(chunk_times_ms)
    return labels, {
        "frames": frames,
        "elapsed_s": elapsed,
        "frames_per_second": rate,
        "mean_ms_per_frame": 1000.0 * elapsed / frames if frames else 0.0,
        "target_fps": fps,
        "estimated_realtime_factor": rate / fps if fps else None,
        "can_keep_up_at_target_rate": bool(rate >= fps),
        "peak_rss_mb_process": _peak_rss_mb(),
        "chunk_size": int(chunk_size),
        "chunk_count": int(len(chunk_times)),
        "chunk_processing_ms": {
            "median": float(np.median(chunk_times)) if len(chunk_times) else 0.0,
            "p95": float(np.quantile(chunk_times, 0.95)) if len(chunk_times) else 0.0,
            "max": float(chunk_times.max()) if len(chunk_times) else 0.0,
        },
    }


def write_frame_report(path, labels, features, control, dead, confidence=None,
                       event_context=None, fps=25):
    """Write a local frame-aligned engineering report; never writes source data."""
    labels = np.asarray(labels)
    control = np.asarray(control)
    dead = np.asarray(dead, dtype=bool)
    n = len(labels)
    if control.shape != (n,) or dead.shape != (n,):
        raise ValueError("labels, control and dead must be frame aligned")
    confidence = np.full(n, np.nan) if confidence is None else np.asarray(confidence)
    if confidence.shape != (n,):
        raise ValueError("confidence must have one value per frame")
    fields = ("home_nearest_distance", "away_nearest_distance",
              "home_second_distance", "away_second_distance",
              "home_nearest_speed", "away_nearest_speed",
              "home_players_nearby", "away_players_nearby", "ball_speed",
              "home_approach_speed", "away_approach_speed", "ball_seen")
    frame_features = {key: np.asarray(features[key]) for key in fields}
    if any(value.shape != (n,) for value in frame_features.values()):
        raise ValueError("feature arrays must be frame aligned")
    # Resolve optional vectors and their fallbacks once, outside the frame loop.
    velocity = (np.asarray(features["ball_velocity"]) if "ball_velocity" in features
                else np.full((n, 2), np.nan))
    direction = (np.asarray(features["ball_direction"]) if "ball_direction" in features
                 else np.full((n, 2), np.nan))
    if velocity.shape != (n, 2) or direction.shape != (n, 2):
        raise ValueError("ball velocity and direction must have shape (frames, 2)")
    timestamps = np.arange(n, dtype=float) / float(fps)
    home_close = frame_features["home_nearest_distance"] <= 2.0
    away_close = frame_features["away_nearest_distance"] <= 2.0
    seen = frame_features["ball_seen"].astype(bool, copy=False)
    contested = home_close & away_close
    loose = seen & ~home_close & ~away_close
    possible_turnover = np.zeros(n, dtype=bool)
    if n > 1:
        home_or_away = np.isin(labels, ("H", "A"))
        possible_turnover[1:] = (home_or_away[1:] & home_or_away[:-1]
                                 & (labels[1:] != labels[:-1]))
    event_context = event_context or {}
    event_present = np.asarray(event_context.get("event_present", np.zeros(n, dtype=bool)))
    set_piece = np.asarray(event_context.get("set_piece", np.zeros(n, dtype=bool)))
    event_team = np.asarray(event_context.get("event_team", np.full(n, -1)))
    if any(value.shape != (n,) for value in (event_present, set_piece, event_team)):
        raise ValueError("event context arrays must be frame aligned")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "timestamp_s", "label", "control", "confidence_m",
                         "home_distance_m", "away_distance_m", "home_second_m",
                         "away_second_m", "home_nearest_speed_mps", "away_nearest_speed_mps",
                         "home_nearby", "away_nearby", "ball_speed_mps",
                         "ball_vx_mps", "ball_vy_mps", "ball_direction_x", "ball_direction_y",
                         "home_approach_mps", "away_approach_mps", "ball_seen", "dead",
                         "contested", "loose", "possible_turnover", "event_present",
                         "set_piece_context", "event_team"))
        for i in range(n):
            writer.writerow((i, timestamps[i], labels[i], int(control[i]), confidence[i],
                             *(frame_features[key][i] for key in fields[:-3]),
                             velocity[i, 0], velocity[i, 1], direction[i, 0], direction[i, 1],
                             frame_features["home_approach_speed"][i],
                             frame_features["away_approach_speed"][i],
                             bool(seen[i]), bool(dead[i]), bool(contested[i]),
                             bool(loose[i]), bool(possible_turnover[i]), bool(event_present[i]),
                             bool(set_piece[i]), int(event_team[i])))


def write_json_report(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
