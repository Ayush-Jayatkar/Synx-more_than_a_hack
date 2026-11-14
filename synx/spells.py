"""Temporal filtering from frame-local control to possession/dead/live labels."""
import numpy as np
from synx.data import AWAY, HOME, NOBODY
FPS = 25
SWITCH_CONTROL_FRAMES = int(0.5 * FPS)
MIN_SPELL_FRAMES = FPS
LETTER = {HOME: "H", AWAY: "A", None: "X"}


class PossessionStateMachine:
    """Incremental state machine with delayed emission for backdated switches.

    ``feed`` emits only finalized frames. A possible switch is buffered until
    it is confirmed or interrupted; this avoids revising already-emitted output
    and keeps results invariant to chunk boundaries. Unconfirmed frames are
    emitted with the prior state (or X before possession is established).
    """

    def __init__(self, switch_frames=SWITCH_CONTROL_FRAMES):
        if not isinstance(switch_frames, (int, np.integer)) or switch_frames < 1:
            raise ValueError("switch_frames must be a positive integer")
        self.switch_frames = int(switch_frames)
        self.state = None
        self.pending = None
        self.pending_count = 0
        self._pending_labels = []
        self._input_frames = 0
        self._output_frames = 0
        self._finished = False

    def _flush_pending(self, label, output):
        if self._pending_labels:
            output.extend([label] * len(self._pending_labels))
            self._pending_labels.clear()
        self.pending = None
        self.pending_count = 0

    def feed(self, ctrl, dead, ball_seen):
        """Consume aligned arrays; return ``(first_output_frame, labels)``."""
        if self._finished:
            raise RuntimeError("cannot feed a finished possession state machine")
        ctrl = np.asarray(ctrl)
        dead = np.asarray(dead, dtype=bool)
        ball_seen = np.asarray(ball_seen, dtype=bool)
        if ctrl.ndim != 1 or dead.shape != ctrl.shape or ball_seen.shape != ctrl.shape:
            raise ValueError("ctrl, dead and ball_seen must be aligned one-dimensional arrays")
        if np.any(~np.isin(ctrl, (NOBODY, HOME, AWAY))):
            raise ValueError("ctrl contains an invalid control code")

        output = []
        first = self._output_frames
        for code, is_dead, seen in zip(ctrl, dead, ball_seen):
            if is_dead:
                self._flush_pending(LETTER[self.state], output)
                output.append("D")
                self.state = None
                continue

            current = int(code) if seen else NOBODY
            if current == NOBODY or current == self.state:
                self._flush_pending(LETTER[self.state], output)
                output.append(LETTER[self.state])
                continue

            if current != self.pending:
                self._flush_pending(LETTER[self.state], output)
                self.pending = current
            self.pending_count += 1
            self._pending_labels.append(current)
            if self.pending_count >= self.switch_frames:
                # The buffered candidate frames are backdated to possession start.
                output.extend([LETTER[current]] * len(self._pending_labels))
                self._pending_labels.clear()
                self.state = current
                self.pending = None
                self.pending_count = 0

        self._input_frames += len(ctrl)
        self._output_frames += len(output)
        return first, np.asarray(output, dtype="<U1")

    def finish(self):
        """Flush any unconfirmed tail using the last established state."""
        if self._finished:
            return self._output_frames, np.empty(0, dtype="<U1")
        first = self._output_frames
        output = []
        self._flush_pending(LETTER[self.state], output)
        self._output_frames += len(output)
        self._finished = True
        if self._output_frames != self._input_frames:
            raise RuntimeError("state machine did not emit one label per input frame")
        return first, np.asarray(output, dtype="<U1")


def label_frames(ctrl, dead, ball_seen, switch_frames=SWITCH_CONTROL_FRAMES):
    """Return H/A/D/X labels. ``dead`` is explicit evidence from integration.

    Sustained team evidence establishes or switches possession; a confirmed
    switch is backdated. NOBODY, unseen ball and brief opposing touches preserve
    established possession but reset pending switch evidence. With no state
    they remain X. Dead frames are D and reset state. No short spell is erased.
    """
    machine = PossessionStateMachine(switch_frames=switch_frames)
    _, emitted = machine.feed(ctrl, dead, ball_seen)
    _, tail = machine.finish()
    return np.concatenate((emitted, tail))


def drop_short_spells(labels, min_len=MIN_SPELL_FRAMES):
    """Optional utility replacing short H/A runs by X; not called automatically."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or min_len < 0:
        raise ValueError("labels must be one-dimensional and min_len nonnegative")
    out = labels.copy()
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]; ends = np.r_[changes, len(labels)]
    for start, end in zip(starts, ends):
        if labels[start] in ("H", "A") and end - start < min_len:
            out[start:end] = "X"
    return out
