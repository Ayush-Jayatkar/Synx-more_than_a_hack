# Workstream 2: Possession Logic & Football Analysis

## Architecture

`data.load()` remains the only ATD parser. It returns metre coordinates aligned to tactical-video frames. `possession_features.build_features()` computes team-level distances, movement, pressure and ball vectors without treating track slots as persistent players. `approaches.py` exposes independent frame-local evidence models. `spells.PossessionStateMachine` maintains temporal state and buffers unconfirmed candidate frames so confirmed changes are backdated without changing results at chunk boundaries. `streaming.py` provides chunked inference and a paced frame-by-frame simulation. `possession.predict_possession()` is the application interface; `run_all.py` writes and validates a submission.

No homography is needed: the ATD coordinates are already in pitch space. No label training or threshold fitting is used. Confidence values are score separations in metres, not calibrated probabilities.

## Frame inputs and outputs

Required tracking arrays from `data.load()`:

| Key | Shape | Units / meaning |
|---|---|---|
| `ball` | `(F, 2)` | x,y metres; missing is NaN |
| `players` | `(F, S, 2)` | slot positions in metres |
| `player_speed` | `(F, S)` | reported speed in m/s |
| `team` | `(S,)` | HOME=0 or AWAY=1 |
| `ball_speed` | `(F,)` | m/s |
| `pitch` | `(length, width)` | metres; used for visible out-of-pitch evidence |

`build_features(track)` returns frame-aligned arrays. Ball vectors (`ball_velocity`, `ball_direction`) are `(F,2)`. Approach speed is the change in team-nearest distance per second; it is weak evidence because the nearest tracklet can change. Player acceleration is not derived across slots because slot identity is not stable.

The prediction API is:

```python
from synx.possession import predict_possession
labels = predict_possession(track, dead=dead_mask, chunk_size=4096)
labels, diagnostics = predict_possession(track, dead=dead_mask,
                                        return_features=True)
```

It returns `(F,)` labels in H/A/D/X. Dead-ball evidence is explicit when supplied. Otherwise, the entry point uses only the published half/in-play windows and visible ball positions beyond the pitch boundary. Missing ball tracking alone is never dead ball. `optional_events` may carry `dead` and frame-aligned context arrays; event context is diagnostic only.

## Independent control approaches

`approaches.distance_based`, `.movement_based`, `.spatial_based`, and `.combined` each return frame-local control codes and a metre-scale confidence score. `compare_controls()` also applies the temporal state model as Approach D. `diagnostics.compare_approaches()` reports predicted label counts, spell durations, confidence distributions, disagreement, missing-ball handling, contested-frame disagreement, transition counts and processing time. These are engineering comparisons, not accuracy measurements.

- A — distance to nearest player per team and a configurable separation margin.
- B — distance plus signed nearest-distance closing evidence and a ball-speed uncertainty margin.
- C — teammate crowding and local opponent pressure.
- D — possession persistence, restart reset, hysteresis and switch backdating.
- E — combined evidence followed by the temporal state machine.

No approach is described as more accurate based on agreement, confidence or counts.

## Streaming and real-time behaviour

`StreamingPossessionEngine.feed(track_chunk, dead)` accepts consecutive chunks and returns a `StreamOutput`. Its label portion can lag input by up to the switch-confirmation window because candidate possession frames remain buffered until confirmed or interrupted. `finish()` flushes the final pending frames. `predict_chunked()` and `iter_track_chunks()` use the same stateful engine. Feature continuity carries the previous raw ball coordinate and team-nearest distances across chunks. Synthetic tests require identical final labels for one-frame, small, and large chunk sizes.

`simulate_realtime(track, dead, pace=True)` feeds one frame at a time and optionally sleeps to the configured 25 FPS schedule. A full paced match takes roughly the match duration; use `frame_limit` for a short smoke run. Offline benchmarking measures frame throughput, mean time per frame, per-chunk latency and process peak RSS where the operating system exposes it. The tracking loader currently materializes the complete match arrays before inference; frame inference itself is chunked.

## Smart Tagging

`events.load_smart_tagging_events()` calls the existing starter parser. `events.align_events()` converts half-open interval boundaries to frame arrays, clips intervals to the video, tracks unknown team values as -1, and counts malformed records. Smart Tagging phases and set-piece ranges are optional context only: they are not possession labels and do not set D or alter control without a separately reviewed integration. The command-line `--smart-tagging` option adds aligned context to the local frame report when the asset is available.

## Parameters

- Control radius: 2.0 m by default.
- Contest margin: 0.4 m by default.
- Teammate crowding penalty: 0.25 m per extra nearby teammate.
- Approach credit: 0.05 m per m/s, bounded to approach evidence of +/-5 m/s.
- Ball-speed uncertainty: 0.02 m per m/s, capped at 1.0 m.
- Possession confirmation: 13 consecutive frames (about 0.5 s at 25 FPS).

These values are explicit heuristics, not fitted or tuned on the competition match labels or video.

## Synthetic testing

```powershell
python tests\test_pipeline.py
python tests\test_possession_features.py
python tests\test_streaming.py
```

Tests use synthetic arrays and temporary synthetic files. They cover control, switches, short genuine spells, deflections, dead-ball resets, missing observations, tracklet disappearance, invalid shapes, non-finite values, empty arrays, large synthetic inputs, event alignment, feature continuity, chunk-boundary continuity, and report row alignment.

## Full local run and outputs

Set `MTAH_DATA_DIR` to the permitted local data bundle, then run from the repository root:

```powershell
$env:MTAH_DATA_DIR = 'D:\path\to\mtah2027'
python run_all.py --chunk-size 4096 --smart-tagging
```

By default this writes `outputs/workstream2/Synx.csv`, validates exactly the configured frame count with `evaluation/validate_submission.py`, and writes local `benchmark.json`, `approach_comparison.json`, `frame_diagnostics.csv`, and `validation.txt` alongside it. `outputs/` and `cache/` are ignored by Git. An existing output is not overwritten without `--overwrite`. Use `--no-comparison` or `--no-frame-report` to skip reports.

Run a short paced simulation without writing a submission:

```powershell
python run_all.py --simulate-realtime --frames 250 --no-comparison
```

Run the complete paced simulation only when you intend to wait through the match duration:

```powershell
python run_all.py --simulate-realtime
```

The validator checks only submission format. Do not run `evaluation/run_evaluation.py` for this workstream.

## Competition compliance and limitations

The local command uses tracking only to generate predictions and engineering diagnostics; it never loads possession ground truth or video. Do not upload tracking, events, caches, reports or predictions to hosted services. Do not commit match data, caches, generated CSVs, or local reports. No real-match accuracy claim is made. Approach speed may be distorted by nearest-tracklet changes, ball velocity can be noisy, automatic event phases can be wrong, and the state model introduces confirmation latency. Smart Tagging remains optional and does not influence labels pending a separate, permitted review.

## Workstream handoff

- Workstream 1 continues to own ATD parsing, cache creation and array schemas. No duplicate parser is introduced.
- Workstream 3 can add richer Smart Tagging/event normalization to `events.py`, validate timestamp/frame alignment on synthetic fixtures, and propose event signals through `optional_events`. Keep event parsing optional and avoid treating phase tags as possession truth.
- Workstream 4 can consume `predict_possession()` or `run_all.predict()` and the label-free diagnostics. Keep generated outputs local.
- Workstream 5 can choose chunk size, offline or paced mode, and output destination without changing the H/A/D/X contract.
