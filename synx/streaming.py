"""Stateful sequential and chunk-based possession prediction."""
from dataclasses import dataclass
import time

import numpy as np

from synx import approaches, possession_features, spells


@dataclass
class StreamOutput:
    """Finalized labels and diagnostics from one input chunk.

    ``input_start`` indexes feature/control arrays. ``label_start`` indexes the
    contiguous subset of labels finalized by this call. A short unresolved
    suffix can remain buffered until a later chunk or ``finish``.
    """
    input_start: int
    label_start: int
    labels: np.ndarray
    control: np.ndarray
    features: dict


class StreamingPossessionEngine:
    """Incremental combined predictor with exact chunk-boundary continuity."""

    def __init__(self, feature_kwargs=None, control_kwargs=None, switch_frames=13,
                 approach="E_combined"):
        self.feature_kwargs = dict(feature_kwargs or {})
        self.control_kwargs = dict(control_kwargs or {})
        self.machine = spells.PossessionStateMachine(switch_frames=switch_frames)
        self.approach = approach
        self.feature_state = None
        self.input_frames = 0
        self.output_frames = 0
        self.finished = False

    def feed(self, track_chunk, dead):
        """Consume one aligned tracking chunk and explicit dead-ball mask."""
        if self.finished:
            raise RuntimeError("cannot feed a finished possession engine")
        ball = np.asarray(track_chunk.get("ball"))
        if ball.ndim != 2:
            raise ValueError("track chunk must contain ball with shape (frames, 2)")
        dead = np.asarray(dead, dtype=bool)
        if dead.shape != (len(ball),):
            raise ValueError("dead must have one boolean per input frame")
        input_start = self.input_frames
        features, self.feature_state = possession_features.build_features(
            track_chunk, previous=self.feature_state, return_state=True,
            **self.feature_kwargs
        )
        control = approaches.run_control(
            self.approach, features, self.control_kwargs
        )["control"]
        label_start, labels = self.machine.feed(
            control, dead, features["ball_seen"]
        )
        self.input_frames += len(ball)
        self.output_frames += len(labels)
        return StreamOutput(input_start, label_start, labels, control, features)

    def finish(self):
        """Emit pending tail frames and close the engine."""
        if self.finished:
            return self.output_frames, np.empty(0, dtype="<U1")
        start, labels = self.machine.finish()
        self.output_frames += len(labels)
        self.finished = True
        if self.output_frames != self.input_frames:
            raise RuntimeError("engine did not produce one label per input frame")
        return start, labels


def iter_track_chunks(track, dead, chunk_size=4096, feature_kwargs=None,
                      control_kwargs=None, switch_frames=13,
                      approach="E_combined"):
    """Yield ``(start_frame, labels, StreamOutput)`` across any chunk size."""
    ball = np.asarray(track["ball"])
    dead = np.asarray(dead, dtype=bool)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball must have shape (frames, 2)")
    if dead.shape != (len(ball),):
        raise ValueError("dead must have one boolean per frame")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    engine = StreamingPossessionEngine(
        feature_kwargs, control_kwargs, switch_frames, approach
    )
    feature_keys = ("ball", "players", "player_speed", "team", "ball_speed")
    for start in range(0, len(ball), chunk_size):
        end = min(len(ball), start + chunk_size)
        chunk = {key: np.asarray(track[key])[start:end] if key != "team"
                 else np.asarray(track[key]) for key in feature_keys}
        output = engine.feed(chunk, dead[start:end])
        if len(output.labels):
            yield output.label_start, output.labels, output
    start, labels = engine.finish()
    if len(labels):
        yield start, labels, None


def predict_chunked(track, dead, chunk_size=4096, feature_kwargs=None,
                    control_kwargs=None, switch_frames=13,
                    approach="E_combined"):
    """Collect a complete deterministic label array from the streaming engine."""
    n = len(track["ball"])
    labels = np.empty(n, dtype="<U1")
    written = np.zeros(n, dtype=bool)
    for start, chunk_labels, _ in iter_track_chunks(
            track, dead, chunk_size, feature_kwargs, control_kwargs,
            switch_frames, approach):
        labels[start:start + len(chunk_labels)] = chunk_labels
        written[start:start + len(chunk_labels)] = True
    if not written.all():
        raise RuntimeError("streaming output contains unfinalized frames")
    return labels


def simulate_realtime(track, dead, frame_limit=None, pace=True, fps=25,
                      feature_kwargs=None, control_kwargs=None,
                      switch_frames=13):
    """Feed one frame at a time, optionally pacing calls to the target frame rate.

    Returns labels and timing diagnostics. ``frame_limit`` is useful for a
    short live-mode smoke run; with ``None`` the whole input is consumed.
    """
    n = len(track["ball"])
    limit = n if frame_limit is None else int(frame_limit)
    if limit < 0 or limit > n:
        raise ValueError("frame_limit must be in 0..number of frames")
    dead = np.asarray(dead, dtype=bool)
    if dead.shape != (n,):
        raise ValueError("dead must have one boolean per input frame")
    engine = StreamingPossessionEngine(
        feature_kwargs, control_kwargs, switch_frames
    )
    keys = ("ball", "players", "player_speed", "team", "ball_speed")
    labels = []
    start_time = time.perf_counter()
    schedule_start = time.perf_counter()
    for frame in range(limit):
        chunk = {key: np.asarray(track[key])[frame:frame + 1] if key != "team"
                 else np.asarray(track[key]) for key in keys}
        result = engine.feed(chunk, dead[frame:frame + 1])
        labels.extend(result.labels.tolist())
        if pace:
            deadline = schedule_start + (frame + 1) / float(fps)
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    _, tail = engine.finish()
    labels.extend(tail.tolist())
    elapsed = time.perf_counter() - start_time
    values = np.asarray(labels, dtype="<U1")
    rate = limit / elapsed if elapsed else float("inf")
    return values, {
        "frames": limit,
        "elapsed_s": elapsed,
        "frames_per_second": rate,
        "target_fps": fps,
        "paced": bool(pace),
        "estimated_realtime_factor": rate / fps if fps else None,
    }
