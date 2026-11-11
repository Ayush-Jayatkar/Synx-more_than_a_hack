# Workstream 1: Data & Tracking Engineering (handoff)

Branch: `feature/workstream1`. Files: `synx/data.py`, `synx/missing_ball.py`, `tests/test_pipeline.py`.

**Goal:** turn the raw tracking files into clean per-frame arrays that every other module can trust.

## Status

| Requirement | Status | Where |
|---|---|---|
| Understand ATD structure | Done | `parse_atd` reads `frame:players:ball` lines; frame_id 1 = video frame 0 |
| Player and ball coordinates | Done | 0-1 coords scaled to metres (105.3 x 68 from `games.py`) |
| Player speeds | Done | `player_speed`, m/s, as reported by the tracking |
| Ball speed | Done | `ball_speed`, m/s from frame-to-frame movement |
| Missing observations and gaps | Done | `NaN` everywhere data is missing; `missing_ball.py` fills ball gaps up to 0.5 s and flags gaps over 3 s |
| Track slots and team mapping | Done | slot -> team from the ATD metadata via the official `team_of_each_track` |
| Reusable loading pipeline | Done | `load()` parses once, then reads a cache |
| Data-quality check | Done | `check()` and `python -m synx.data` |
| Tests | Done | synthetic data only, no match files |

## How to use it

```python
from synx import data
track = data.load()
track["ball"]          # (199319, 2) metres, NaN = ball not seen
track["players"]       # (199319, 46, 2) metres, NaN = slot empty
track["player_speed"]  # (199319, 46) m/s
track["ball_speed"]    # (199319,) m/s, NaN where either frame lacks the ball
track["team"]          # (46,) 0 = home (Barcelona), 1 = away (Athletic)
track["pitch"]         # (105.3, 68.0)
```

Row i is video frame i, the same as submission row i.

```bash
python -m synx.data            # data check over the graded frames
python tests/test_pipeline.py  # self-checks on synthetic data
```

The first `load()` parses the raw file (a minute or two) and saves `cache/BAR-ATH_tracking.npz`. Later
calls read the cache in seconds. If the raw or metadata file changes (size or modified time), the cache
is rebuilt automatically. `cache/` is gitignored: it is derived from the match data and must never be
committed or shared.

## Data check

`python -m synx.data` prints, over the graded frames only:

- `ball_seen_pct`: share of frames with a ball position
- `longest_ball_gap_s`: longest stretch without the ball
- `mean_tracked_players`: average number of tracked people per frame
- `frames_over_22_tracked_pct`: frames where the tracker reports more than 22 people (duplicates)
- `home_slots`, `away_slots`: how many slots belong to each team

Read these yourself. They describe the match data, so do not paste them into any AI tool or chat.
Compare them with the published figures in the official `DATA.md` (ball visible on 59.3% of graded
frames, about 20.5 tracked people per graded frame). A large mismatch means a parsing bug.

## Decisions

- **Slots are tracklets, not players.** 46 slots for 22 players; they appear, vanish and swap. Every
  feature is per team, never per slot.
- **Duplicates.** When the tracker reports one player twice on the same team, the nearest-player distance
  per team is unchanged, so no correction is needed. A person tracked as both teams cannot be told apart
  from two real players; that case is only measured (`frames_over_22_tracked_pct`), not fixed.
- **Ball speed uses the raw ball**, not the gap-filled one, so filled frames never produce made-up speeds.
- **Missing stays NaN.** No zeros or placeholders, so a missing value can never look like a real position.

## Open items for the next owner

1. Ball speed is noisy where the ball tracking jitters. A short median filter (3-5 frames) may help;
   pick the window from physics (max kick speed ~35 m/s), not from scores.
2. Cross-team duplicates are not removed. Possible rule: two opposite-team slots within ~0.3 m for
   several frames are one person; keep the one with the longer track history.
3. Ball height is not used. The raw ball group may carry more than x, y; check the official schema
   (`BAR-ATH_tactical_atd.xml`) before relying on it.
4. Run `python -m synx.data` once on the real data and confirm the numbers match `DATA.md`.

## Rules for this workstream

- Match data is input only. Never tune a parameter by looking at the ground truth or the video.
- Never paste raw rows, data-check output or screenshots into AI tools (ChatGPT, Claude, Gemini, Copilot).
- Never commit `data/` or `cache/`.
