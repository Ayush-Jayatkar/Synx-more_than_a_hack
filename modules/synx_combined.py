"""Synx H/A/D probabilities with explicit, chunk-safe feature context."""
import numpy as np

from interfaces import PossessionModule
from synx import approaches, pitch, possession_features
from synx.ball_possession import enhanced_scores
from synx.data import AWAY, HOME, FPS
from synx.possession_features import DEFAULT_CONTROL_RADIUS_M


class SynxCombined(PossessionModule):
    """Convert Workstream 2 evidence to normalized H/A/D probabilities.

    ``predict_proba`` processes a complete sequence from absolute frame zero.
    ``predict_proba_chunk`` is pure across calls: callers explicitly pass the
    prior feature state and absolute frame start, and receive the next feature
    state. Temporal possession smoothing remains in Synx's state machine.
    """

    def __init__(self, feature_kwargs=None, control_kwargs=None,
                 probability_temperature_m=0.75, max_team_probability=0.98,
                 reach_uncertainty_m=0.5):
        if (not np.isfinite(probability_temperature_m)
                or probability_temperature_m <= 0):
            raise ValueError("probability_temperature_m must be finite and positive")
        if (not np.isfinite(max_team_probability)
                or not 0.5 < max_team_probability < 1.0):
            raise ValueError("max_team_probability must be between 0.5 and 1.0")
        if not np.isfinite(reach_uncertainty_m) or reach_uncertainty_m <= 0:
            raise ValueError("reach_uncertainty_m must be finite and positive")
        self.feature_kwargs = dict(feature_kwargs or {})
        self.control_kwargs = dict(control_kwargs or {})
        self.probability_temperature_m = float(probability_temperature_m)
        self.max_team_probability = float(max_team_probability)
        self.reach_uncertainty_m = float(reach_uncertainty_m)

    @staticmethod
    def _fallback(n):
        probabilities = np.zeros((n, 3), dtype=np.float64)
        probabilities[:, :2] = 0.5
        return probabilities

    @staticmethod
    def _dead_mask(track, ball, n, frame_start=0):
        """Return validated explicit/default dead evidence."""
        if "dead" in track:
            raw_dead = np.asarray(track["dead"])
            if raw_dead.shape != (n,):
                raise ValueError("track['dead'] must have shape (n_frames,)")
            if raw_dead.dtype.kind == "b":
                return raw_dead
            if (raw_dead.dtype.kind in "iuf"
                    and np.isfinite(raw_dead).all()
                    and np.isin(raw_dead, (0, 1)).all()):
                return raw_dead.astype(bool)
            raise ValueError("track['dead'] must contain only booleans or 0/1")
        if "pitch" in track:
            return (pitch.ball_out_of_pitch(np.asarray(ball, dtype=float), track["pitch"])
                    | ~pitch.in_play_window(n, start_frame=frame_start))
        return np.zeros(n, dtype=bool)

    def predict_proba(self, track):
        """Return whole-sequence probabilities in ``[H, A, D]`` column order."""
        probabilities, _ = self.predict_proba_chunk(
            track, previous_features=None, frame_start=0
        )
        return probabilities

    def predict_proba_chunk(self, track, previous_features, frame_start):
        """Return ``(probabilities, next_features)`` for one consecutive chunk.

        ``previous_features`` is the state returned by the preceding call;
        ``frame_start`` is the chunk's absolute match-frame index. No state is
        retained on this model instance. For exact whole/chunk equivalence,
        chunks must be contiguous and callers must pass both values.
        """
        if not isinstance(track, dict) or "ball" not in track:
            raise ValueError("track must be a mapping containing 'ball'")
        raw_ball = np.asarray(track["ball"])
        if raw_ball.ndim == 0:
            raise ValueError("track['ball'] must have a frame dimension")
        n = int(raw_ball.shape[0])
        if not isinstance(frame_start, (int, np.integer)) or frame_start < 0:
            raise ValueError("frame_start must be a nonnegative absolute frame index")
        if frame_start == 0 and previous_features is not None:
            raise ValueError("the first chunk must start without previous_features")
        if frame_start > 0:
            required = ("frame_index", "ball", "home_nearest_distance",
                        "away_nearest_distance")
            if (not isinstance(previous_features, dict)
                    or any(key not in previous_features for key in required)
                    or previous_features["frame_index"] != frame_start - 1):
                raise ValueError(
                    "chunk context must end at frame_start - 1; pass prior chunk state"
                )
        fallback = self._fallback(n)
        local_track = dict(track)
        dead = self._dead_mask(local_track, raw_ball, n, frame_start)
        fallback[dead] = (0.0, 0.0, 1.0)
        if n == 0:
            return fallback, previous_features
        try:
            # Rebuild the first ball-speed value from explicit previous-frame
            # context. This also handles callers that calculate speed per chunk.
            if n and previous_features is not None:
                speed = np.asarray(local_track["ball_speed"], dtype=float).copy()
                prior_ball = np.asarray(previous_features.get("ball"), dtype=float)
                current_ball = np.asarray(local_track["ball"], dtype=float)
                if prior_ball.shape == (2,):
                    # The first chunk velocity is only defined when both
                    # endpoints are visible. Do not retain a chunk-local
                    # speed across a missing previous-frame observation.
                    if np.isfinite(current_ball[0]).all():
                        if np.isfinite(prior_ball).all():
                            speed[0] = np.hypot(*(current_ball[0] - prior_ball)) * FPS
                        else:
                            speed[0] = np.nan
                    else:
                        speed[0] = np.nan
                    local_track["ball_speed"] = speed
            features, next_features = possession_features.build_features(
                local_track, previous=previous_features, return_state=True,
                **self.feature_kwargs
            )
            if n:
                next_features["frame_index"] = int(frame_start + n - 1)
            hn = np.asarray(features["home_nearest_distance"], dtype=float)
            an = np.asarray(features["away_nearest_distance"], dtype=float)
            if hn.shape != (n,) or an.shape != (n,):
                raise ValueError("feature arrays must align with the input frames")

            hscore, ascore, margin = enhanced_scores(features, **self.control_kwargs)
            delta = ascore - hscore
            radius = self.control_kwargs.get(
                "radius", self.feature_kwargs.get(
                    "control_radius", DEFAULT_CONTROL_RADIUS_M
                )
            )
            d_min = np.minimum(hn, an)
            reach_logit = np.clip(
                (radius - d_min) / self.reach_uncertainty_m, -40.0, 40.0
            )
            reach = 1.0 / (1.0 + np.exp(-reach_logit))
            reach[~np.isfinite(d_min)] = 0.0

            # E supplies the signed continuous evidence. A/B/C/E are run with
            # their existing code paths; their agreement changes reliability,
            # not the team score, avoiding repeated addition of correlated cues.
            control_options = dict(self.control_kwargs)
            codes = [approaches.run_control(name, features, control_options)["control"]
                     for name in ("A_distance", "B_movement", "C_spatial", "E_combined")]
            expected = np.where(delta > 0, HOME, AWAY)
            expected[~np.isfinite(delta) | (delta == 0)] = -1
            aligned = np.stack([code == expected for code in codes], axis=0)
            agreement = aligned.mean(axis=0)
            evidence_reliability = 0.25 + 0.75 * agreement

            seen = np.asarray(features["ball_seen"], dtype=bool)
            contest_logit = np.clip(
                (np.abs(delta) - margin) / self.probability_temperature_m,
                -40.0, 40.0,
            )
            contest_clarity = 1.0 / (1.0 + np.exp(-contest_logit))
            q = np.clip(reach * evidence_reliability * contest_clarity * seen,
                        0.0, 1.0)
            q[~np.isfinite(delta)] = 0.0
            speed_uncertainty = np.asarray(margin, dtype=float)
            temperature = self.probability_temperature_m + speed_uncertainty
            log_odds = np.clip(
                np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0) / temperature,
                -40.0, 40.0,
            )
            p_home_given_control = 1.0 / (1.0 + np.exp(-log_odds))
            home_p = 0.5 + q * (p_home_given_control - 0.5)
            home_p = np.clip(home_p, 1.0 - self.max_team_probability,
                             self.max_team_probability)

            result = np.zeros((n, 3), dtype=np.float64)
            result[:, 0] = home_p
            result[:, 1] = 1.0 - home_p
            if dead.any():
                result[dead] = (0.0, 0.0, 1.0)
            if not np.isfinite(result).all() or np.any(result < 0):
                raise ValueError("probability output was non-finite or negative")
            return result, next_features
        except (KeyError, TypeError, ValueError, IndexError, AttributeError,
                FloatingPointError):
            raise
