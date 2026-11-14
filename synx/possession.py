"""End-to-end possession prediction over loaded tracking arrays."""
import numpy as np

from synx import approaches, pitch, possession_features, spells, streaming


def default_dead_mask(track):
    """Build D only from published half windows and visible out-of-pitch evidence."""
    ball = np.asarray(track["ball"])
    dead = np.zeros(len(ball), dtype=bool)
    if "pitch" in track:
        dead |= pitch.ball_out_of_pitch(ball, track["pitch"])
        dead |= ~pitch.in_play_window(len(ball))
    return dead


def predict_possession(track, optional_events=None, dead=None,
                       return_features=False, control_kwargs=None,
                       feature_kwargs=None, spell_kwargs=None,
                       chunk_size=4096, approach="E_combined"):
    """Predict one H/A/D/X label per tracking frame.

    ``optional_events`` can contain an explicit frame-aligned ``dead`` mask
    and/or event context arrays. Event context is exposed diagnostically only;
    automatic phase tags are not treated as possession labels. If no dead mask
    is supplied, D uses only published half windows and visible pitch exits.
    Missing ball tracking alone never implies dead ball.
    """
    if not isinstance(track, dict) or "ball" not in track:
        raise ValueError("track must be a data.load() tracking dictionary")
    ball = np.asarray(track["ball"])
    if ball.ndim != 2 or ball.shape[1:] != (2,):
        raise ValueError("track ball must have shape (frames, 2)")
    n = len(ball)
    if optional_events is not None and not isinstance(optional_events, dict):
        raise ValueError("optional_events must be a mapping")
    if dead is None and optional_events is not None:
        dead = optional_events.get("dead")
    dead_mask = default_dead_mask(track) if dead is None else np.asarray(dead, dtype=bool)
    if dead_mask.shape != (n,):
        raise ValueError("dead must have one boolean per frame")
    spell_kwargs = dict(spell_kwargs or {})
    switch_frames = spell_kwargs.pop("switch_frames", spells.SWITCH_CONTROL_FRAMES)
    if spell_kwargs:
        raise TypeError("Unknown spell options: {}".format(sorted(spell_kwargs)))
    labels = streaming.predict_chunked(
        track, dead_mask, chunk_size=chunk_size,
        feature_kwargs=feature_kwargs, control_kwargs=control_kwargs,
        switch_frames=switch_frames, approach=approach,
    )
    if not return_features:
        return labels
    features = possession_features.build_features(track, **(feature_kwargs or {}))
    evidence = approaches.run_control(approach if approach != "D_temporal" else "A_distance",
                                      features, control_kwargs)
    diagnostics = dict(features)
    diagnostics.update({"control": evidence["control"],
                        "confidence": evidence["confidence"],
                        "dead": dead_mask})
    if optional_events is not None:
        for key, value in optional_events.items():
            if key != "dead":
                array = np.asarray(value)
                if array.ndim and len(array) == n:
                    diagnostics["event_" + key] = array
    return labels, diagnostics
