# Team Synx - Technical Report

More Than a Hack 2027. Football possession from tracking data. This report describes what the code in this
repository does. It makes no accuracy or score claim. File names in brackets point to the source of each statement.

## 1. Problem and objective

The task is to label every frame of the tactical video (BAR-ATH, 199,319 frames, 25 FPS) with one of four labels:

- `H` home team has possession
- `A` away team has possession
- `D` dead ball
- `X` abstain

The output is a CSV with columns `frame,label`, one row per video frame [`run_all.py`, `games.py`]. Our objective is a
deterministic, explainable pipeline that uses tracking data only, with no training on the match.

## 2. Overall approach

The method is rule-based. No model is trained and no parameter is fitted. The production path is:

```text
synx/data.py  ->  synx/possession_features.py  ->  synx/approaches.py (E_combined)
                                                        |
synx/possession.py + synx/stoppages.py (dead mask, tagged) -> synx/spells.py (state machine) via synx/streaming.py
                                                        |
                                                  run_all.py -> Synx.csv
```

1. Member 1 data pipeline: parse the tracking file into per-frame arrays in metres (`synx/data.py`).
2. Features: per frame, the nearest and second-nearest player of each team, ball motion, approach speed (`synx/possession_features.py`).
3. Frame-level control evidence: Approach E_combined decides, frame by frame, whether the home team, the away team or nobody controls the ball (`synx/approaches.py`, `synx/ball_possession.py`).
4. Dead-ball mask: half windows and ball beyond the pitch lines (`synx/possession.py`, `synx/pitch.py`). `run_all.py` uses `--dead-ball tagged` by default, which adds tracking evidence and Smart Tagging set pieces (`synx/stoppages.py`, section 11). `--dead-ball basic` keeps only the first two.
5. State machine: turns frame-level control into possession spells with hysteresis (`synx/spells.py`).
6. Chunked engine: runs steps 2 to 5 chunk by chunk (`synx/streaming.py`). `run_all.py` calls `streaming.predict_chunked`.

`possession.predict_possession` is the same engine exposed as a library call and gives the same labels (both use 13 switch frames).

Not used by `run_all.py`: `interfaces.py` and `modules/` (registry, nearest-player, probability adapter) are optional
offline helpers. From `starter/`, only `team_of_each_track` (`starter/baseline.py`) and `KICKOFF_VIDEO_FRAMES`
(`starter/load_smart_tagging.py`) are imported; the nearest-player baseline is not run. `synx/missing_ball.py` and `synx/tracking_diagnostics.py` are also
not imported by the production path.

## 3. Assets and their purposes

| Asset | Used for predictions | Role |
|---|---|---|
| Tactical tracking, ATD (`BAR-ATH_tactical_atd.txt` and `_atd.xml`) | Yes | Ball and player positions and speeds; slot-to-team mapping from the metadata [`synx/data.py`] |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | Yes, in the default `tagged` mode (dead-ball only) | Prediction input in `tagged` mode, the `run_all.py` default: set-piece tags mark where a stoppage is looked for [`synx/stoppages.py`]. Team tags and phase codes are not used. With `--smart-tagging` it also fills extra columns of a diagnostics CSV, which does not change labels [`synx/events.py`, `synx/diagnostics.py`]. `--dead-ball basic` and `tracking` do not read it. The starter kit recommends set pieces as dead-ball evidence and the team confirmed it is an allowed input. The XML is match data and is never committed. |
| Homography | No | Not read by any production code. ATD is already in pitch coordinates. |
| Tactical and broadcast video | No | Not read. |
| Public ground truth (`evaluation/gt/`) | No | Not read by `run_all.py`. Kept locally only, not in Git. |

The two kickoff frame numbers used for the in-play window (11185 and 113742) are constants in
`starter/load_smart_tagging.py` (`KICKOFF_VIDEO_FRAMES`), described there as published values. The Smart Tagging file
itself is not opened for that. In the diagnostics CSV, `event_team` is always -1 because `run_all.py` passes no
team mapping to `events.align_events`.

## 4. Member 1 data engineering and preprocessing

`synx/data.py` is the only tracking parser used by the pipeline.

- Input line format: `frame_id:x,y,speed;x,y,speed;...:ball_x,ball_y,...`. Lines that do not have three
  colon-separated parts are skipped. An unreadable number, or a player block that does not divide into 46 slots,
  raises `ValueError` naming the frame id. Frame ids outside 1..199,319 are ignored.
- Output arrays, one row per video frame: `ball` (n,2), `players` (n,46,2), `player_speed` (n,46) as reported in m/s,
  `team` (46,), `ball_speed` (n,), `pitch`. Missing values are NaN. Arrays are float32.
- Slot-to-team: the first team in the metadata is home, the second is away (`team_of_each_track` in
  `starter/baseline.py`). On the local data bundle this gives 46 slots: 24 home and 22 away. Slots are tracklets, not people. They appear,
  vanish and swap, so the pipeline uses them only by team.
- Ball height: `data.load` adds `ball_z` only if the metadata declares a `z` ball channel. The local metadata for this match declares no ball
  height channel, and the pipeline does not use `ball_z` in any case.
- Ball speed: `hypot(delta position) * 25` between consecutive frames. It is NaN on frame 0 and wherever either frame
  has no ball.
- Cache: the parsed arrays are saved to `cache/BAR-ATH_tracking.npz`. A stamp of `CACHE_VERSION` (2) plus the size and
  modification time of the XML and TXT files is stored with it. The cache is rebuilt when the stamp differs or the
  file cannot be read. The cache is written to a temporary file and renamed. On 2026-11-17 a fresh re-parse of the local source gave
  exactly the cached `ball`, `players`, `player_speed`, `ball_speed` and `team`.
- The loader holds the whole match in memory. Inference after loading is chunked.

## 5. Coordinate conversion and tracking alignment

- The raw ATD coordinates are 0 to 1 fractions of the pitch. They are multiplied by (105.3, 68.0) m
  [`games.py` `pitch`, `synx/data.py` `parse_atd`]. No rotation or direction normalisation is applied. Home and away
  are handled symmetrically, so attacking direction is not needed.
- Frame alignment: ATD `IdFrame` counts from 1, video frames from 0. Cache frame `i` is `IdFrame` `i + 1`
  (`frame = int(parts[0]) - 1` in `parse_atd`). Frames with no ATD record stay NaN. On the local bundle the ATD covered all 199,319 video frames. The
  submission is indexed by this same frame number.
- The per-half offsets in `games.py` (`offsets`, `halves`) describe a separate counter. The pipeline uses them only
  through `half_video_blocks` to get the last in-play frame of each half (see section 11).
- `games.py` finds the data folder from the `MTAH_DATA_DIR` environment variable. The default is `./data`.

## 6. Feature engineering and movement calculations

Computed in `synx/possession_features.py` (`build_features`), per frame, for each team separately:

- `*_nearest_distance`: distance in metres from the ball to the nearest tracked slot of that team. A slot with no
  position is ignored. If the ball or all slots are missing the value is NaN.
- `*_second_distance`: distance to the second-nearest slot of that team.
- `*_nearest_speed`: the reported speed (m/s) of that nearest slot.
- `*_players_nearby`: number of slots of that team within the control radius (2.0 m).
- `distance_advantage`: away nearest minus home nearest (positive means home is closer).
- `ball_speed`: from `synx/data.py`; negative or non-finite values become NaN.
- `ball_velocity` and `ball_direction`: from the ball position change between consecutive frames, times 25 for velocity,
  normalised for direction. NaN if either frame has no ball. For the first frame of a chunk the previous chunk's last
  ball position is used.
- `*_approach_speed`: closing speed of the team's nearest distance, `(previous nearest distance - current nearest distance) * 25`
  (m/s, positive means the team is getting closer). See section 13 for when it is rejected.
- `ball_seen`: both ball coordinates are finite.

Ball direction and velocity are written to the diagnostics CSV. The control evidence (section 7) does not use them.

## 7. Main Workstream 2 algorithm

Frame-level control uses Approach E_combined [`synx/ball_possession.py` `enhanced_scores`, `enhanced_control`,
called from `synx/approaches.py` `combined`]. For each team a score is computed. Lower is stronger evidence:

```text
score = nearest_distance
        - 0.25 * support            support = max(nearby - 1, 0), at least 1 if the second-nearest is within 3.0 m, clipped to 0..2
        + 0.15 * clip(opponent_nearby, 0, 3)
        - 0.05 * clip(approach_speed, 0, 5)              (NaN counts as 0)
        + 0.02 * clip(max(ball_speed - nearest_player_speed, 0), 0, 20)
margin = 0.4 + min(0.02 * ball_speed, 1.0)               (ball_speed capped at 50 m/s)
```

A team controls the ball in a frame when its nearest player is within 2.0 m and its score plus the margin is still
lower than the other team's score. If neither team meets this, the frame is "nobody". A NaN nearest distance (ball unseen, or no player of that team) makes the comparison false, so the frame is also
"nobody". Other NaN inputs fall back to neutral values: no teammate support, approach speed 0, no speed penalty, and
margin 0.4 when ball speed is NaN (`synx/ball_possession.py`).

Other approaches (A distance, B movement, C spatial) exist in `synx/approaches.py` for comparison. The production
engine runs E_combined only (`approach="E_combined"` in `synx/streaming.py`).

Football reasons for the terms: a player close to the ball is likely in control; teammates nearby support the
carrier; opponents nearby pressure him; a player closing on the ball gets a small credit; a fast ball is less likely
to be under control, and a nearest player much slower than the ball is unlikely to be playing it.

## 8. State machine and hysteresis

`synx/spells.py` (`PossessionStateMachine`) reads, per frame, the control code, the dead flag and `ball_seen`.

- State: current possessing team, or none (labelled X).
- Frames where control equals the current state, or is "nobody", keep the current label. Possession therefore
  carries through passes between teammates and through loose balls.
- When control shows the other team (or any team while the state is none), that team becomes a candidate. The
  candidate is confirmed after 13 control frames in a row (`SWITCH_CONTROL_FRAMES`, 0.52 s). Any frame of "nobody", or
  of the old team, cancels the candidate. The count then starts again.
- On confirmation, the buffered candidate frames are relabelled to the new team (backdated), so the spell starts at its
  first control frame. A candidate that is cancelled keeps the old label for its frames.
- Because of backdating, a label is emitted only once it is final. The engine holds back at most the unconfirmed
  frames. At the end of the input the remaining buffered frames take the last established state (`finish`).
- A candidate can bridge one unseen frame (`PENDING_GAP_FRAMES` = 1). The unseen frame does not count as control. A
  second consecutive unseen frame cancels the candidate.
- No minimum spell length is applied. `spells.drop_short_spells` exists but is never called.

## 9. Home and Away decisions

`H` or `A` is written when the state is the home or away team (state 0 or 1 in `synx/data.py`: `HOME`, `AWAY`). The team
is set only by a confirmed 13-frame run of E_combined control for that team. Before the first confirmed run, and
after every `D` or long missing-ball period, the state is none and the label is `X`. A single opposing touch shorter
than 13 consecutive control frames does not change the label. A real turnover shorter than that is also not
detected.

## 10. Contested and loose ball

- Contested: both teams have a player within 2.0 m. E_combined gives control to a team only if its score beats the
  other by more than the margin. Otherwise the frame is "nobody" and the current possession label stays.
- Loose: no player within 2.0 m of the ball. The frame is "nobody", so the label stays with the last team that had
  possession. If there is none, it stays `X`.
- A challenge by an opponent changes the label only if that team then wins control on 13 consecutive frames.
- `contested`, `loose` and `possible_turnover` columns in the diagnostics CSV are computed in `synx/diagnostics.py` from
  the same 2.0 m radius, for inspection only.

## 11. Dead-ball detection

`possession.default_dead_mask` [`synx/possession.py`, `synx/pitch.py`] marks a frame as dead when either:

1. It is outside the in-play window. The window of each half starts at the published kickoff frame (11185 for the
   first half, 113742 for the second) and ends at the last tracked frame of that half (90,651 and 190,197, from the
   half spans and offsets in `games.py`). Frames before the first kickoff, between the halves and after the second
   half are `D`.
2. The ball is seen and is more than 0.5 m beyond a touchline or goal line (`OUT_MARGIN_M` in `synx/pitch.py`), using
   the 105.3 x 68.0 m pitch.

A missing ball never makes a frame dead. A dead frame resets the possession state, so after a restart the label is `X`
until a team has 13 consecutive control frames. If a candidate switch was pending when a dead frame arrived, its frames
get the previous label (no backdating). In `basic` mode, set pieces and stoppages with the ball inside the pitch
lines are not detected as dead.

`run_all.py --dead-ball basic|tracking|tagged` selects the evidence (`synx/stoppages.py`). The command-line default is
`tagged` and `python run_all.py` writes the final submission (SHA-256
`67c29b1642fd7b5e348160502679cd66b397e999ceeeeb286c06a5e02b9055a0`). Library calls (`run_all.predict`, `run_all.run`,
`possession.predict_possession`) with no dead mask still use the basic mask. The mask is built on the whole match before chunked inference, so labels do not depend on
chunk size. A missing ball alone never becomes `D`. Team tags and phase codes are not used.

### 11.1 `basic`

The two rules above. `python run_all.py --dead-ball basic` reproduces the previous submission, kept as a fallback (SHA-256
`c46a268a6bed3024405498d236b922e67a0121b66d2d5f006e16c1d77c0285fb`).

### 11.2 `tracking`

`basic` plus two tracking-only rules:

- Still ball: the ball is seen and its filtered speed is below 0.5 m/s for at least 2 s (50 frames). In open play the
  ball is almost never motionless for 2 s; a ball placed for a free kick, goal kick or corner is.
- Out-of-play bridge: unseen frames that directly follow a frame where the ball was seen out of play stay `D`, for at
  most 30 s. The ball is being retrieved off camera, and a restart normally follows within 30 s.

### 11.3 `tagged` (default)

`tracking` plus Smart Tagging set pieces. A tag window is a fixed 35 s (5 s for kick offs), so it is never copied
across as `D`. Tracking finds the stoppage inside it:

- Evidence is the ball out of play, or the ball at rest for at least 1 s (a set-piece tag confirms a restart nearby).
  The first evidence frame from 10 s before to 15 s after the window start begins the stoppage.
- `D` runs until the restart: the ball seen inside the pitch at 5 m/s or more for at least 10 frames (0.4 s), which is
  faster than a player carrying the ball and long enough to exclude one noisy frame.
- If no restart is found, only the evidence frames in the window become `D`. If there is no evidence, the set piece is
  skipped.
- `Kick Off (After Goal)`: `D` from the last ball-out frame within 120 s before the window to the restart. If there is
  no restart, up to the window start.
- `Kick Off (Start)` is ignored: the in-play window already covers it.

Malformed events are counted as skipped and do not stop the run.

## 12. Missing ball and X

- Unseen ball: features are NaN, so the frame gives no control evidence.
- If a candidate switch is pending, see section 8.
- Otherwise the previous label is held for up to 25 consecutive unseen frames (`MISSING_BALL_GRACE_FRAMES`, 1 s). From
  the 26th unseen frame the state is cleared and the label is `X`.
- After the ball reappears with no state, the label stays `X` until a team has 13 consecutive control frames.
- `X` is also used before the first possession of the match period and after every dead-ball period.
- Missing ball is never turned into `D`.

## 13. Tracking discontinuity handling

Slots are tracklets, so the "nearest player" can jump when a slot appears, disappears or swaps. Approach speed is
protected against this in `_approach_speed` [`synx/possession_features.py`]:

- The change in nearest distance between two frames cannot exceed (ball displacement) + (the larger of the two nearest
  players' displacements). Both are measured as speeds. A change larger than `ball_speed + max(nearest_speed now,
  nearest_speed before)` (+1e-6) is treated as a discontinuity and the approach speed becomes NaN, which counts as 0 in the
  score. This means a tracklet jump never earns approach credit.
- Approach speed is also NaN if any of the distances or speeds are missing or negative.
- The check uses the previous frame's values, carried across chunk boundaries.
- Players with no position in a frame are left out of the distance calculation. Slots are not filtered by position
  (for example, bench slots on the sideline are not excluded).
- Opposite-team slots lying on top of each other are not detected or removed in the production path.
  `synx/tracking_diagnostics.py` can report them but is not used by `run_all.py`.

## 14. Chunked prediction and reproducibility

- `run_all.py` processes the match in chunks of 4096 frames by default (`--chunk-size`).
- Each chunk gets features and control, then the state machine. Carried between chunks: the last ball position, last
  nearest distances and speeds (feature state), and the state machine (state, candidate, buffered frames, missing-ball
  counter).
- Ball speed is computed over the whole track in `synx/data.py` before chunking.
- Nothing is random. The labels do not depend on the chunk size. Local verification run (2026-11-17): chunk sizes 256, 1024, 4096 and the
  whole track gave 0 labels different from the 4096 run, and two fresh runs gave identical SHA-256 hashes. The synthetic
  chunk-invariance tests are in `tests/test_streaming.py`. To repeat on the real data, see README section 10.
- Libraries are pinned (`numpy==2.1.3`, `pandas==2.2.3`).

## 15. Validation and testing performed

The results below come from a local run on 2026-11-17 with the official data bundle. Section 10 of the README has the commands to repeat each check.

Validation was about correctness and format. We make no accuracy claim, and aggregate label shares are not reported as accuracy.

- `python -m pytest tests/ -v`: all 130 tests pass, on the pinned stack (numpy 2.1.3, pandas 2.2.3) and on
  numpy 2.5.4 / pandas 3.0.6. They use synthetic data only. Each test file except `tests/test_module_architecture.py`
  (pytest only) also runs as a script (`python -B tests/<file>.py`). `tests/test_streaming.py`
  (`test_long_match_chunk_sizes_match_offline_engine`) runs a 5,000-frame synthetic match at chunk sizes 1, 25, 100,
  512 and 4096 and checks it against the whole-track run and `possession.predict_possession`. A short ball gap
  (over the 512 boundary), a ball gap longer than 1 s (over the 4096 boundary) and dead runs (one over the 2000 boundary) are included.
- `tests/test_stoppages.py` (synthetic) covers the `--dead-ball` rules.
- The checks above show the pipeline runs correctly and the output has the right format. They do not measure
  prediction accuracy. No parameter was chosen from any score.
- `--dead-ball tagged` on the full match (no ground truth): dead frames 46,528 (`basic`) to 57,928. Tracking rules
  added 8,551 (still ball 2,172 frames in 22 runs; out-of-play bridge 6,379). Smart Tagging added 2,849 more. Of 57 set
  pieces considered (the 2 `Kick Off (Start)` tags excluded), 35 were located with a restart, 1 of 4 after-goal kick
  offs was used and 21 were skipped for lack of stoppage evidence. Label changes against `basic`: 12,174 frames (X to
  D 9,192; H to D 1,297; A to D 911; H to X 325; A to X 449). The official validator and `verify_synx.py` reported OK,
  chunk sizes 256, 1024, 4096 and the whole track gave 0 mismatches, and two fresh runs gave the same SHA-256
  `67c29b1642fd7b5e348160502679cd66b397e999ceeeeb286c06a5e02b9055a0`.
- One pre-registered public five-minute sample check per version (`evaluation/run_evaluation.py --gt public`), never
  used to tune anything; the thresholds were fixed before it was run:

  | Version | macroF1 | D F1 | spellF1 | transition F1 (+/-25 frames) |
  |---|---|---|---|---|
  | `basic` | 0.5511 | 0.125 | 0.4588 | 0.158 |
  | `tagged` | 0.6697 | 0.481 | 0.4588 | 0.158 |

  spellF1 and transition F1 did not move because the scorer forward-fills predicted `X` inside graded spans and treats
  `X` and `D` alike for alive/dead. `X` frames that followed an existing `D` were already scored as `D` for spells and
  transitions, and the new `D` mostly replaced such `X`. Five minutes (17 possessions) is not representative of the
  full match, so there is no full-match accuracy claim.
- `git diff --check`: clean.
- `python evaluation/validate_submission.py outputs/workstream2/Synx.csv`: OK. 199,319 rows, frames 0..199,318, labels only
  H, A, D, X.
- `python verify_synx.py --input ... --diagnostics ...`: OK.
- Chunk invariance and run-to-run identity, as in section 14.
- Re-parse of the source tracking file matched the cache (section 4).
- The `--dead-ball basic` output was byte-identical on both library stacks above
  (SHA-256 `c46a268a6bed3024405498d236b922e67a0121b66d2d5f006e16c1d77c0285fb`).
- Integration fix: `spells.SWITCH_CONTROL_FRAMES` was `int(0.5 * 25)` = 12 while the streaming engine used by
  `run_all.py` had a literal default of 13, so `possession.predict_possession` and `run_all.py` disagreed on 1,443
  frames. The constant is now 13 and the `synx/streaming.py` state-machine defaults (switch, pending gap, missing-ball grace) are
  read from `synx/spells.py`. Diagnostic-only helpers in `synx/approaches.py` and `synx/diagnostics.py` keep a literal 13. The two calls now
  agree on every frame, and the `run_all.py` output did not change.

## 16. Limitations and future work

- No full-match accuracy measurement was made. The only score is the single public five-minute sample check in
  section 15. Parameters are fixed heuristics chosen from football reasoning. They were not fitted or tuned on the
  ground truth or video.
- Control is based on distance in 2D. There is no ball height channel, so a ball passing over a player looks like control.
- Possession needs 13 consecutive control frames to start or switch. Short real turnovers are missed.
- In `--dead-ball basic` mode, set pieces, injuries and other stoppages with the ball in play area are not marked `D`.
- `--dead-ball tracking|tagged` (`tagged` is the default): still ball can also fire on a goalkeeper holding the ball still. In the real-data
  run 21 of 57 set pieces were skipped for lack of stoppage evidence. A restart with a slow ball (below 5 m/s) is not
  detected, so `D` then ends at the evidence frames. Only 1 of 4 after-goal kick offs found a ball-out frame.
- After 25 unseen frames the label is `X`, even if play continued with the same team.
- Slot swaps, duplicate slots on one person, and bench slots are not filtered.
- The in-play window uses hard-coded kickoff frames and the half spans from `games.py`.
- Future work: use ball height if it becomes available, and filter duplicate and off-pitch slots.

## 17. Installation and reproduction

Python 3.12 is the version named in `requirements.txt`. The 2026-11-17 validation run used Python 3.13.7. Full instructions are in
[README.md](README.md). In short:

```bash
python -m venv .venv            # then activate it
pip install -r requirements.txt
# put the official data bundle in ./data (data/tactical/BAR-ATH_tactical_atd.xml, _atd.txt and _smart_tagging.xml) or set MTAH_DATA_DIR
python run_all.py --output Synx.csv --overwrite
python evaluation/validate_submission.py Synx.csv
```

Default output path without `--output` is `outputs/workstream2/Synx.csv`. The default run uses `--dead-ball tagged`. `--dead-ball basic` needs only the two ATD files.

## Parameter table

| Parameter | Value | File | Football reason |
|---|---|---|---|
| Frame rate | 25 FPS | `synx/data.py` (`FPS`) | Tracking and video rate |
| Pitch size | 105.3 x 68.0 m | `games.py` | Converts 0-1 coordinates to metres |
| Control radius | 2.0 m | `synx/ball_possession.py` (`CONTROL_RADIUS_M`), `synx/possession_features.py` | A player can only control a ball within about a stride |
| Contest margin | 0.4 m | `synx/ball_possession.py` | The winner must be clearly closer than the other team |
| Teammate support credit | 0.25 m per supporter, supporters clipped to 0..2 | `synx/ball_possession.py` | Nearby teammates back up a carrier |
| Second-player support radius | 3.0 m | `synx/ball_possession.py` | A second teammate this close counts as support |
| Opponent pressure penalty | 0.15 m per opponent, clipped to 0..3 | `synx/ball_possession.py` | Pressure makes control less certain |
| Approach credit | 0.05 m per m/s, only positive, clipped to 5 m/s | `synx/ball_possession.py` | A player closing on the ball is about to play it |
| Speed mismatch penalty | 0.02 m per m/s of deficit, clipped to 20 | `synx/ball_possession.py` | A player much slower than the ball is not playing it |
| Ball speed margin | 0.02 m per m/s, margin increase capped at 1.0 m, ball speed capped at 50 m/s | `synx/ball_possession.py` | A fast ball is rarely under control |
| Switch confirmation | 13 frames (0.52 s) | `synx/spells.py` (`SWITCH_CONTROL_FRAMES`), `synx/streaming.py` | A touch or deflection lasts a few frames; control lasts longer |
| Pending-switch gap | 1 frame | `synx/spells.py` (`PENDING_GAP_FRAMES`) | Bridges a single dropped ball frame |
| Missing-ball grace | 25 frames (1 s) | `synx/spells.py` (`MISSING_BALL_GRACE_FRAMES`) | Short tracking drop-outs usually do not end possession |
| Out-of-pitch margin | 0.5 m | `synx/pitch.py` (`OUT_MARGIN_M`) | Ball radius plus tracking noise |
| Kickoff frames | 11185, 113742 | `starter/load_smart_tagging.py` | Start of each half |
| Last in-play frames | 90,651 and 190,197 | `games.py` (`halves`, `offsets`) | End of each half |
| Still-ball speed | 0.5 m/s | `synx/stoppages.py` (`STILL_SPEED_MS`) | Below this the ball is at rest, allowing for tracking jitter |
| Still-ball duration | 50 frames (2 s) | `synx/stoppages.py` (`STILL_MIN_FRAMES`) | A placed ball, not a ball in open play |
| Still-ball duration near a set-piece tag | 25 frames (1 s) | `synx/stoppages.py` (`TAGGED_STILL_MIN_FRAMES`) | 1 s is enough when a tag confirms a restart nearby |
| Out-of-play bridge | 750 frames (30 s) | `synx/stoppages.py` (`MAX_OUT_BRIDGE_FRAMES`) | A restart normally follows the ball leaving the pitch within 30 s |
| Restart speed | 5.0 m/s | `synx/stoppages.py` (`RESTART_SPEED_MS`) | A throw or kick moves the ball faster than a player carrying it |
| Restart duration | 10 frames (0.4 s) | `synx/stoppages.py` (`RESTART_MIN_FRAMES`) | A single noisy frame is not a restart |
| Set-piece look-back | 250 frames (10 s) | `synx/stoppages.py` (`TAG_LOOKBACK_FRAMES`) | The stoppage can begin shortly before the tagged window |
| Set-piece evidence span | 375 frames (15 s) | `synx/stoppages.py` (`TAG_EVIDENCE_FRAMES`) | The stoppage must begin within the first 15 s of the window |
| After-goal look-back | 3,000 frames (120 s) | `synx/stoppages.py` (`GOAL_LOOKBACK_FRAMES`) | A kick off after a goal follows the goal within 2 minutes |
| Cache version | 2 | `synx/data.py` (`CACHE_VERSION`) | Forces a rebuild if the cached arrays change |
| Default chunk size | 4096 frames | `run_all.py` | Memory use only; labels do not change |
