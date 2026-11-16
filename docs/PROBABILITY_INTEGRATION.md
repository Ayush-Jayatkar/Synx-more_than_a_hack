# Synx frame probability adapter

## Integration boundary

`modules.synx_combined.SynxCombined.predict_proba(track)` accepts the loaded tracking mapping and returns one `float64` row per frame in **H, A, D** order. It reuses `synx.possession_features.build_features()`; it does not parse the raw ATD file. It does not emit final labels, smooth probabilities, impose minimum spells, or write submission files. `predict_proba_chunk(track, previous_features, frame_start)` supports explicit context for consecutive chunks: callers pass the prior returned feature state and absolute first-frame index. No temporal state is retained on the model instance. The central runner owns final transition handling.

Member 5's central runner and its loader/feature interface were not present in this checkout, so direct integration against that runner could not be performed. The adapter currently follows the documented `data.load()` array schema. A caller may supply a frame-aligned boolean `track["dead"]`; otherwise the adapter applies the existing pitch and in-play window rules when `pitch` is available. Without either source, it does not infer dead ball from missing tracking.

## Evidence and probability construction

For each team, the adapter reuses `synx.ball_possession.enhanced_scores()` (Approach E): nearest distance, bounded teammate support (including a second teammate within 3 m), opponent pressure, bounded positive approach credit, and ball/player speed-mismatch cost. It also runs the existing A, B, C and E control paths. Vote agreement changes evidence reliability; their scores are not added a second time, avoiding duplicate counting of correlated cues. These are heuristic physical scales, not fitted values.

Let `delta = away_score - home_score`, `m` be Approach E's contest margin,
and `d_min` the nearer team distance. The conditional Home probability is
`p = sigmoid(delta / (T + m))`, with base temperature `T = 0.75 m`.
Evidence strength is `q = reach * reliability * clarity * ball_seen`, where
`reach = sigmoid((radius - d_min) / 0.5 m)`,
`clarity = sigmoid((abs(delta) - m) / T)`, and
`reliability = 0.25 + 0.75 * (A/B/C/E vote agreement fraction)`.
For live frames, `H = 0.5 + q * (p - 0.5)`, `A = 1 - H`, and `D = 0`;
H/A are capped at 0.98. Weak reach, contested scores, disagreement, missing
ball, or non-finite evidence pull H/A toward 0.5. High ball speed broadens
uncertainty through E's margin and the logistic temperature, but never creates
D or independently proves loose possession. Explicit dead frames, or
existing pitch/out-of-play evidence when pitch metadata is available, yield
`[0, 0, 1]`.

At a chunk boundary, the explicit context carries the previous absolute frame
index, ball coordinate, and team-nearest distances. The adapter uses these to
compute first-frame ball motion and approach evidence exactly as in
whole-track processing, including after a missing previous ball. `frame_start`
anchors pitch-window calculations to absolute match frames.

Because the runner contract has only H/A/D columns, abstention is represented as low H/A separation; the runner must apply its own abstention threshold. These values are probabilities normalized mathematically, but are not empirically calibrated probabilities or accuracy claims.

## Limitations and verification

Tracking slots are transient tracklets, so nearest-player approach speed and speed mismatch are weak cues. A/B/C/E votes are correlated frame-local heuristics, not independent evidence. Ball direction is not used to infer control, and the adapter has no temporal memory: a sustained opponent gain must be handled by the central runner's transition logic. Pitch-out and half-window dead-ball semantics follow the existing application behavior; missing `pitch` means no automatic dead-ball inference unless an explicit dead mask is provided.

Run the synthetic-only verification from the repository root:

```powershell
python tests\test_probability_module.py
python tests\test_pipeline.py
python tests\test_possession_features.py
python tests\test_streaming.py
```

The tests cover stable H/A evidence, transitions in both directions, teammate passes, fast ball movement, approach without takeover, contested/loose situations, missing gaps, dead-ball separation, goalkeeper-like distribution, coordinate noise, discontinuity, invalid/empty input, chunk slicing, normalization, and input immutability. They establish deterministic software behavior only; they do not measure match accuracy. No competition ground truth or video is used.
