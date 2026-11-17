"""Shared interface and probability helpers for optional possession modules.

Modules consume the tracking dictionary produced by :mod:`synx.data` and return
one ``[H, A, D]`` probability row per video frame. Temporal smoothing, spell
handling, abstention (``X``), and chunking stay in the existing Synx pipeline.
"""

from __future__ import annotations

import numpy as np

HOME, AWAY = 0, 1
LABELS = np.array(["H", "A", "D"])
N_CLASSES = len(LABELS)


class PossessionModule:
    """Base contract for independent possession probability modules."""

    name = "base"

    def predict_proba(self, track: dict) -> np.ndarray:
        """Return probabilities with shape ``(n_frames, 3)`` in H/A/D order."""
        raise NotImplementedError


def _as_probability_matrix(values, *, who="?"):
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(
            f"[{who}] expected a two-dimensional (n_frames, {N_CLASSES}) array, "
            f"got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"[{who}] probabilities must all be finite")
    return probabilities


def normalize(probabilities):
    """Normalize nonnegative H/A/D scores row-wise.

    All-zero rows become uniform probabilities. Invalid or negative values are
    rejected so callers cannot silently turn broken module output into labels.
    """
    scores = _as_probability_matrix(probabilities, who="normalize")
    if np.any(scores < 0):
        raise ValueError("[normalize] scores must be nonnegative")
    # Scale before summing: summing finite values such as 1e308 can overflow.
    row_max = scores.max(axis=1, keepdims=True, initial=0.0)
    scaled = np.zeros_like(scores)
    np.divide(scores, row_max, out=scaled, where=row_max > 0)
    totals = scaled.sum(axis=1, keepdims=True)
    result = np.full_like(scores, 1.0 / N_CLASSES)
    np.divide(scaled, totals, out=result, where=totals > 0)
    return result


def check_output(probabilities, n_frames, who="?"):
    """Validate a module's H/A/D probability matrix and return its array view."""
    output = _as_probability_matrix(probabilities, who=who)
    if output.shape != (n_frames, N_CLASSES):
        raise ValueError(
            f"[{who}] expected shape ({n_frames}, {N_CLASSES}), got {output.shape}"
        )
    if np.any(output < -1e-9):
        raise ValueError(f"[{who}] negative probability")
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"[{who}] rows must sum to 1")
    return output


def blend_probabilities(outputs, weights=None):
    """Blend compatible H/A/D probability matrices with a convex average.

    Equal weights are used by default. Explicit weights are normalized before
    blending; this helper is not wired into the final submission pipeline.
    """
    try:
        outputs = list(outputs)
    except TypeError as error:
        raise ValueError("module outputs must be an iterable of probability matrices") from error
    matrices = []
    for i, output in enumerate(outputs):
        matrix = _as_probability_matrix(output, who=f"module {i}")
        matrices.append(check_output(matrix, matrix.shape[0], who=f"module {i}"))
    if not matrices:
        raise ValueError("at least one module output is required")
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("all module outputs must have the same shape")

    if weights is None:
        normalized_weights = np.full(len(matrices), 1.0 / len(matrices))
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (len(matrices),):
            raise ValueError(f"expected {len(matrices)} blend weights")
        if not np.isfinite(normalized_weights).all() or np.any(normalized_weights < 0):
            raise ValueError("blend weights must be finite and nonnegative")
        max_weight = normalized_weights.max(initial=0.0)
        if max_weight <= 0:
            raise ValueError("blend weights must have a positive sum")
        normalized_weights = normalized_weights / max_weight
        normalized_weights /= normalized_weights.sum()

    stacked = np.stack(matrices, axis=0)
    return np.tensordot(normalized_weights, stacked, axes=(0, 0))
