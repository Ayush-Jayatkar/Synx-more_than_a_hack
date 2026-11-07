# Team Synx - Phase 1 Report

## 1. Approach
Fixed rule-based pipeline, no machine learning, no training:
1. Load ball and player tracking in metres (`synx/data.py`).
2. Fill short ball gaps by linear interpolation (`synx/missing_ball.py`).
3. Per frame, a team *controls* the ball if its nearest player is within reach (`synx/ball_possession.py`).
4. Mark dead ball: outside the two halves, ball beyond a line, or ball unseen for a long time (`synx/pitch.py`, `synx/missing_ball.py`).
5. A state machine turns control into spells with hysteresis, back-dating each switch to the first control frame; short spells become X (`synx/spells.py`).

**Why this rather than something else.** Rules rather than a learned model, because we have no licensed labelled possession data from other matches, and the provided match may not be used for training or tuning. A rule-based pipeline has a handful of parameters, each with a stated football reason, and is fully deterministic: the same data always gives the same file. It improves on the official baseline (nearest player plus majority smoothing) by requiring the ball to be within reach before anyone controls it, by keeping possession through passes and loose balls, and by detecting dead ball explicitly instead of treating every missing ball as dead.

## 2. Assets used and dropped
| Asset | Used? | How / why not |
|---|---|---|
| Tracking (`BAR-ATH_tactical_atd.txt`) | Yes | Ball and player positions; slot -> team from the ATD metadata |
| Homography (`BAR-ATH_tactical_homography.csv`) | No (v1) | Tracking is already in pitch coordinates and the tactical homography is valid on every frame |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | No (v1) | Candidate for set-piece dead-ball detection in v2 |
| Tactical video | No | Only watched to understand the possession definition |
| Broadcast folder | No | Not aligned to the tactical video. |

## 3. Missing and noisy data
- Ball gaps of at most 0.5 s are filled linearly.
- Gaps longer than 3 s are treated as dead ball.
- Gaps in between carry the current possession; if none is established, the frame is X.
- Missing players are ignored for that frame. Tracklet ID switches don't matter: only the team of each slot is used.

## 4. Possession rules
- **D**: outside the two halves (published kickoff frames and half ends), ball more than 0.5 m beyond a touchline or goal line, or ball unseen for more than 3 s. D resets possession.
- **H/A starts or switches**: the other team must control the ball on 12 frames (0.5 s), with no gap longer than 0.5 s between them. The new spell starts at its first control frame. A teammate's pass keeps the spell; a brief opponent touch does not switch it.
- **X**: live ball with no established possession (e.g. right after a restart), and any spell shorter than 1 s.

Where we drew the line:
- **Passes and loose balls**: while nobody is within reach, possession stays with the last team that had it. A pass between teammates is one spell.
- **Challenges and deflections**: an opponent touching the ball is not a turnover. Possession changes only when the opponent keeps the ball within reach for 0.5 s. If both teams are within reach, the nearer player counts.
- **Restarts**: dead ball ends the spell. After a restart, possession starts once a team has controlled the ball for 0.5 s. The frames before that are X.

## 5. Parameters
Every value is set from football reasoning or licensed external data, never from this match's ground truth or video.

| Parameter | Value | Reason |
|---|---|---|
| Control radius | 2.0 m | One stride plus tracking noise between ball and player centre |
| Switch evidence | 12 frames (0.5 s) | A touch or deflection lasts a few frames; real control is several touches over about half a second |
| Evidence expiry | 0.5 s gap | Unrelated opponent touches seconds apart should not add up to a turnover |
| Short-gap fill | 0.5 s | Over half a second a ball moves roughly straight |
| Long-gap dead | 3 s | Longer than a long pass stays in the air |
| Out margin | 0.5 m | Ball radius ~0.11 m plus tracking noise; the whole ball must cross the line |
| Min spell | 1 s | Shorter possessions are not credible spells, so abstain |

## 6. Limitations
- Long occlusions during live play become D.
- Set pieces and stoppages with the ball still on the pitch are not detected (Smart Tagging not used yet).
- Hard threshold on control radius, with no confidence score.

## 7. Reproducibility
See README "Reproduce Synx.csv".

## 8. Training data
None. No labelled data of any kind was used: the pipeline has no trained components. No hand labels, ground truth or human-produced data about the provided game were used.
