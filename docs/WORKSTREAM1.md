# Workstream 1: Data & Tracking Engineering

Files owned: `synx/data.py`, `synx/tracking_diagnostics.py`, `synx/missing_ball.py`, `tests/test_data.py`.

**Goal:** turn the raw tracking files into clean per-frame arrays that every other module can trust.
Workstream 2 (`feature/possession-logic`) is the authoritative possession algorithm; nothing here
changes its logic, state machine or CSV output.

## Public data arrays

`data.load()` returns a dict. Row i is video frame i (= submission row i); `n` = 199,319.

| Key | Shape | Type | Units | Missing value | Present |
|---|---|---|---|---|---|
| `ball` | (n, 2) | float32 | metres (x along length, y along width) | NaN | always |
| `players` | (n, slots, 2) | float32 | metres | NaN where the slot is empty | always |
| `player_speed` | (n, slots) | float32 | m/s as reported by the tracking | NaN | always |
| `ball_speed` | (n,) | float32 | m/s, raw frame-to-frame | NaN on frame 0 and wherever either frame lacks the ball | always |
| `team` | (slots,) | int8 | 0 = home (Barcelona), 1 = away (Athletic) | none | always |
| `pitch` | tuple | float | (105.3, 68.0) metres | none | always |
| `ball_speed_filtered` | (n,) | float32 | m/s, median-filtered | NaN exactly where `ball_speed` is NaN | only with `ball_speed_window` |
| `ball_z` | (n,) | float32 | as reported, not rescaled | NaN | only if the metadata declares a ball `z` channel |

The six "always" keys are unchanged from the previous version, so Workstream 2 code keeps working.
Coordinates come in normalised 0-1 and are multiplied by the pitch size. Slots are tracklets, not
people: they appear, vanish and swap, so use them only by team.

## Configuration

| Option | Default | Effect |
|---|---|---|
| `load(ball_speed_window=None)` | None | 3 or 5 adds `ball_speed_filtered` |
| `load(use_cache=True)` | True | False always parses the raw file and writes no cache |
| `load(cache_dir=...)` | `cache/` | where the parsed arrays are stored |
| `cross_team_duplicates(max_distance_m=0.3)` | 0.3 m | how close two opposite-team slots must be |
| `cross_team_duplicates(min_frames=25)` | 25 frames (1 s) | how long they must stay that close |

## Features

### 1. Ball-speed noise filter

`data.filter_ball_speed(speed, window)` is a centred median over 3 or 5 frames. Each frame uses only
neighbours from its own unbroken run of valid speeds, so it never smooths across a missing-ball gap
of any length and never creates a speed inside one. The raw `ball_speed` is kept unchanged; the
filtered array is a separate key. Window sizes come from the signal, not from scores: a one- or
two-frame jitter spike is removed by a 3- or 5-frame median, while a real kick (which changes speed
and then holds it) is kept.

### 2. Cross-team duplicate diagnostics

`tracking_diagnostics.cross_team_duplicates(players, team)` reports pairs of one home and one away
slot that stay within 0.3 m of each other for at least 1 s in a row (a frame where either is missing
breaks the run). Each candidate records `home_slot`, `away_slot`, `home_team`, `away_team`,
`start_frame`, `end_frame` (inclusive), `n_frames`, `median_distance_m`, `confidence` and `reason`.
Confidence is `high` when the run lasts at least 2 s and the median distance is at most 0.15 m,
otherwise `medium`. It never deletes or edits tracks, and it is separate from the loader.

Thresholds come from body size: two real players' tracked centres stay about a shoulder-width
(0.4-0.5 m) apart even in a tight duel, so centres within 0.3 m for a full second are very unlikely
to be two different people.

```bash
python -m synx.tracking_diagnostics   # writes outputs/workstream1/duplicate_candidates.csv (local, gitignored)
```

### 3. Ball height

The EPTS metadata XML declares the order of the ball fields as `<BallChannelRef channelId="..."/>`
entries; this is the same schema the kloppy EPTS reader builds its parser from.
`data.ball_channels(meta_path)` reads that list. If it contains `z`, `load()` adds `ball_z`, taken
from that position in each ball record, with NaN where missing and no rescaling (x and y are
normalised to the pitch, but the schema defines no such normalisation for height). If there is no
`z` channel, there is no `ball_z` key, and nothing is estimated or invented.

**Status for this match:** the official `DATA.md` lists only `ball_x` and `ball_y` for the ball,
so height is probably not available. That has not been checked against the real file here, because
the match data is not opened in AI tools. Check it locally:

```bash
python -m synx.data --schema
```

It prints only the channel names, not data values. Until that says `z` is present, treat ball height
as unavailable.

### 4. Cache

- The first `load()` parses the raw file (a minute or two) and saves `cache/BAR-ATH_tracking.npz`.
  Later calls read it in seconds.
- The cache is rebuilt automatically when the raw file or the metadata file changes (size or
  nanosecond modified time), or when the cache format version changes.
- A corrupt, empty or half-written cache is treated as missing and rebuilt.
- Writes go to a temporary file first and are then renamed, so an interrupted run never leaves a
  broken cache.
- `cache/` and `outputs/` are gitignored: they are derived from the match data and must never be
  committed or shared.

### 5. Input checks

- Lines that are not `frame:players:ball` records are skipped, as in the official baseline.
- An unreadable number or frame id, or a player block that does not match the slot count, raises
  `ValueError` naming the frame id.
- Frame ids outside 1..n are ignored. An empty file gives all-NaN arrays.

## Data check

`python -m synx.data` prints, over the graded frames only: ball-seen %, longest ball gap, mean
tracked players, % of frames with more than 22 tracked, slots per team, and whether ball height is
available. Read these yourself and compare them with the official `DATA.md` (ball visible on 59.3%
of graded frames, about 20.5 tracked people per graded frame). Do not paste them into any AI tool.

## Test results

Run on synthetic data only (no match data, no ground truth), Python 3.11, numpy 2.1.3, pandas 2.2.3:

| Suite | Result |
|---|---|
| `tests/test_data.py` (Workstream 1, new) | 21 / 21 pass |
| `tests/test_pipeline.py` (Workstream 2) | 21 / 21 pass |
| `tests/test_possession_features.py` | 13 / 13 pass |
| `tests/test_streaming.py` | 13 / 13 pass |
| `tests/test_probability_module.py` | pass ("All probability module tests passed") |
| `git diff --check` | clean |

`tests/test_data.py` covers: frame indexing and alignment, coordinate conversion, array and team
dimensions, speed from known positions, NaN handling, the speed filter (noise spike, gaps, edges,
invalid windows), long missing-ball gaps, cache creation, rebuild on raw and metadata changes,
corrupt-cache recovery, raw vs cached equality, ball height present and absent, empty and malformed
input, and duplicate detection (detection with ranges, persistence, threshold, opposite teams only,
runs broken by missing frames, inputs not modified).

```bash
python tests/test_data.py
```

## Known limitations

1. Ball height availability for this match is unconfirmed until `python -m synx.data --schema` is
   run locally.
2. Duplicate pairs are only reported. Choosing which slot to drop is left to downstream modules.
3. Duplicate detection checks every home-away slot pair over the full match; that takes longer than
   loading but is a one-off diagnostic.
4. The speed filter only removes short spikes; it cannot fix longer stretches where the ball track
   follows the wrong object.
5. Nothing here has been run on the real match data in this workstream; run `python -m synx.data`
   and `python -m synx.tracking_diagnostics` locally once.

## Rules for this workstream

- Match data is input only. Never tune a parameter on the ground truth, the video or prediction scores.
- Never paste raw rows, data-check output or screenshots into AI tools (ChatGPT, Claude, Gemini, Copilot).
- Never commit `data/`, `cache/` or `outputs/`.
