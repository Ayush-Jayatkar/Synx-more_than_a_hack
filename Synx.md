# Synx – Football Possession Prediction

## 1. Project Overview

Synx is a football possession prediction system that identifies which team has possession of the ball at every frame of a match using player and ball tracking data.

It analyses player positions, movement, distance from the ball, teammate support and opponent pressure to estimate possession. It also detects dead-ball situations and maintains consistent possession states throughout the match.

The system processes the BAR-ATH tactical match containing 199,319 frames at 25 FPS.

Each frame is assigned one of four labels:

| Label | Description |
|---|---|
| `H` | Home team possession |
| `A` | Away team possession |
| `D` | Dead-ball situation |
| `X` | Unknown or uncertain possession |

The final output is a CSV file containing `frame,label`, with one prediction for every frame.

## 2. System Architecture

The system follows a sequential processing pipeline:

```text
          Tracking Data
                |
                v
       Data Preprocessing
                |
                v
        Feature Extraction
                |
                v
     Possession Estimation
                |
                v
       Dead-Ball Detection
                |
                v
   Possession State Management
                |
                v
         Chunk Processing
                |
                v
            CSV Output
```

### Main Components

| Component | Function |
|---|---|
| `data.py` | Loads and preprocesses tracking data |
| `possession_features.py` | Extracts player and ball movement features |
| `ball_possession.py` | Calculates possession scores and control decisions |
| `approaches.py` | Implements possession approaches, including E_combined |
| `possession.py` | Handles possession prediction and basic dead-ball detection |
| `pitch.py` | Defines pitch dimensions and boundary checks |
| `stoppages.py` | Detects dead-ball situations |
| `spells.py` | Maintains possession states and transitions |
| `streaming.py` | Processes frames in chunks |
| `run_all.py` | Executes the complete prediction pipeline |

## 3. Data Processing and Preprocessing

The system uses tactical tracking data containing player positions, player speeds and ball coordinates.

The data loader in `synx/data.py` converts the input into structured arrays:

- `ball`: Ball coordinates for each frame.
- `players`: Coordinates of 46 player tracking slots.
- `player_speed`: Reported player speed in metres per second.
- `team`: Mapping of each player slot to the home or away team.
- `ball_speed`: Speed calculated from consecutive ball positions.
- `pitch`: Pitch dimensions used for coordinate conversion.

The match contains 46 tracking slots: 24 home and 22 away. These slots represent tracking identities rather than permanent player identities, as they may appear or disappear during the match.

### Coordinate Conversion

Raw tracking coordinates are normalised between 0 and 1 and converted into metres using the pitch dimensions:

- Pitch length: 105.3 metres
- Pitch width: 68.0 metres

Tracking frame identifiers start at 1, whereas video frames start at 0. The system converts them into matching zero-based frame indices.

Missing coordinates are represented as `NaN` to prevent unavailable tracking information from being treated as valid positions.

### Tracking Cache

Parsed tracking arrays are stored locally in:

`cache/BAR-ATH_tracking.npz`

The cache is rebuilt when the source-file information or cache version changes. This avoids repeatedly parsing the complete tracking files.

## 4. Feature Extraction

Feature extraction is implemented in `synx/possession_features.py`.

For each frame, the system calculates the following features separately for both teams:

- **Nearest-player distance:** Distance between the ball and the closest tracked player.
- **Second-nearest distance:** Distance to the second-closest player.
- **Nearest-player speed:** Speed of the closest tracked player.
- **Players nearby:** Number of teammates within the 2.0-metre control radius.
- **Distance advantage:** Difference between home and away nearest-player distances.
- **Approach speed:** Rate at which the nearest player moves closer to the ball.
- **Ball speed and direction:** Calculated using consecutive ball positions.
- **Ball visibility:** Indicates whether valid ball coordinates are available.

### Tracking Discontinuity Handling

Tracking slots may appear, disappear or change between frames. This can create sudden changes in the nearest-player distance.

The system checks whether these changes can be explained by ball and player movement. If a change is too large to be physically consistent, the approach-speed feature is rejected for that frame.

This prevents tracking-slot changes from incorrectly influencing possession scores.

## 5. Possession Prediction – E_combined

Possession is estimated using the **E_combined approach**, implemented through `synx/ball_possession.py` and `synx/approaches.py`.

A possession score is calculated independently for the home and away teams. A lower score represents stronger evidence of control.

The approach considers five factors:

1. **Ball distance:** Distance between the nearest player and the ball.
2. **Team support:** Nearby teammates who can support possession.
3. **Opponent pressure:** Proximity of opposing players.
4. **Approach speed:** How quickly a player moves towards the ball.
5. **Speed mismatch:** Difference between player and ball movement.

The combined scoring calculation is:

```text
score = nearest_distance
        - 0.25 * support
        + 0.15 * opponent_nearby
        - 0.05 * approach_speed
        + 0.02 * speed_mismatch

margin = 0.4 + min(0.02 * ball_speed, 1.0)
```

The individual terms are clipped to their configured limits before being combined.

### Control Decision

A team is assigned control when:

- Its nearest player is within the 2.0-metre control radius.
- Its score is better than the opposing team's score by the required margin.

If neither team satisfies these conditions, the frame is treated as having no confirmed control.

The production pipeline uses E_combined. Other approaches, including distance, movement and spatial approaches, are available in `synx/approaches.py`.

## 6. Possession State Management

Frame-level possession estimates can fluctuate because of tackles, deflections, loose balls and tracking noise.

To maintain consistent possession labels, the system uses a state machine implemented in `synx/spells.py`.

### Possession Confirmation

A new team must demonstrate control for 13 consecutive frames before a possession change is confirmed.

At 25 FPS, this represents approximately 0.52 seconds.

When a change is confirmed, the buffered candidate frames are assigned to the new team from the first candidate frame. If the candidate is interrupted, the previous possession state is retained.

The system also allows one unseen frame during a pending possession change.

### Contested and Loose Balls

- **Contested ball:** When both teams have players close to the ball, their possession scores are compared. If neither team has sufficient evidence, no new control is assigned.
- **Loose ball:** When no player is within the control radius, the previous possession label is retained.
- **Possession change:** An opposing team must establish control for 13 consecutive frames before the label changes.

This prevents short touches and temporary tracking variations from causing unnecessary possession switches.

## 7. Dead-Ball Detection

Dead-ball detection identifies situations where normal gameplay is interrupted.

The system uses pitch boundaries, ball movement, tracking information and Smart Tagging set-piece events.

Three detection modes are available:

| Mode | Detection |
|---|---|
| `basic` | Match in-play periods and ball-out-of-pitch detection |
| `tracking` | Basic detection plus still-ball and out-of-play tracking evidence |
| `tagged` | Tracking detection with Smart Tagging set-piece events |

The default mode is `tagged`.

### 7.1 Basic Detection

A frame is marked as dead when:

- It falls outside the defined in-play period.
- The visible ball is more than 0.5 metres beyond the pitch boundary.

The in-play periods use the defined kickoff frames and half-match boundaries.

### 7.2 Tracking-Based Detection

The tracking mode adds two additional detection methods:

- **Still ball:** Detects a visible ball moving below 0.5 m/s for at least 50 frames (2 seconds).
- **Out-of-play bridge:** Extends an out-of-play period across missing-ball frames for up to 30 seconds.

### 7.3 Smart Tagging-Supported Detection

The `tagged` mode uses Smart Tagging set-piece events to identify situations where a stoppage is expected.

Tracking evidence is used to locate the stoppage and its end. A stoppage can be identified through the ball leaving the pitch or remaining at rest.

The dead-ball period continues until the ball moves back into the pitch at the configured restart speed for the required number of frames.

If sufficient evidence is unavailable, the event is skipped rather than automatically marking the entire tag window as dead.

Team tags and phase codes are not used by the prediction logic.

A dead-ball frame resets the possession state. After play restarts, the label remains `X` until a team establishes control for 13 consecutive frames.

## 8. Missing Ball and Uncertain Possession

Missing ball coordinates are not treated as dead-ball evidence.

The system handles missing tracking as follows:

- The previous possession label is retained for up to 25 consecutive unseen frames.
- From the 26th consecutive unseen frame, the possession state is cleared and the label becomes `X`.
- When the ball reappears, a team must establish control for 13 consecutive frames before `H` or `A` is assigned again.
- Before the first confirmed possession and after dead-ball periods, the label is also `X` until control is established.

This separates uncertain tracking from an actual stoppage in play.

## 9. Chunked Processing and Reproducibility

The complete match is loaded and parsed before inference. Prediction is then performed in chunks to manage processing while preserving state between sections of the match.

The default chunk size is 4096 frames.

The streaming engine carries the following information between chunks:

- Previous ball position.
- Previous nearest-player distances and speeds.
- Current possession state.
- Pending possession-change candidate and buffered frames.
- Missing-ball counter.

The dead-ball mask is prepared for the complete match before chunked inference. This ensures that dead-ball decisions do not change at chunk boundaries.

The implementation is deterministic. The same input and configuration produce the same output.

Chunk-size checks were performed using different chunk sizes, and repeated runs produced identical predictions.

## 10. Technical Parameters

| Parameter | Value |
|---|---:|
| Match frame rate | 25 FPS |
| Number of frames | 199,319 |
| Pitch dimensions | 105.3 × 68.0 m |
| Player tracking slots | 46 |
| Home tracking slots | 24 |
| Away tracking slots | 22 |
| Ball-control radius | 2.0 m |
| Possession switch confirmation | 13 frames |
| Pending-switch gap | 1 frame |
| Missing-ball grace period | 25 frames |
| Out-of-pitch margin | 0.5 m |
| Still-ball speed threshold | 0.5 m/s |
| Still-ball duration | 50 frames |
| Out-of-play bridge | 750 frames |
| Default chunk size | 4096 frames |

## 11. Technical Validation

The implementation was checked for correct execution, consistent state handling and submission-file structure.

The following checks were performed:

- Automated test suite: 130 tests passed.
- Stoppage detection rules tested with synthetic inputs.
- Chunked predictions compared across multiple chunk sizes.
- Repeated runs produced identical output.
- Tracking source re-parsed and compared with cached arrays.
- `git diff --check` completed without errors.
- Submission validator accepted the generated CSV.

The generated file contains 199,319 rows, with frame numbers from 0 to 199318 and labels restricted to `H`, `A`, `D` and `X`.

These checks verify implementation behaviour and output format; they do not represent a full-match accuracy measurement.

## 12. System Limitations

- Ball height is not available in the tracking data, so control is estimated using two-dimensional positions.
- Possession changes shorter than the 13-frame confirmation period may not be detected.
- A goalkeeper holding a stationary ball can sometimes resemble a stoppage.
- Some set-piece events may not have enough tracking evidence to identify the complete dead-ball period.
- A slow restart below the configured restart-speed threshold may not be detected.
- After 25 consecutive unseen ball frames, the system changes the label to `X`.
- Player tracking-slot changes, duplicate slots and off-pitch bench slots are not fully filtered.

## 13. Technical Contributions

| Responsibility | Technical Contribution |
|---|---|
| Data Engineering | Tracking-file parsing, coordinate conversion, team mapping, array generation and caching |
| Feature Engineering | Player-to-ball distances, movement calculations, approach speed and team-level features |
| Possession Logic | E_combined scoring, control decisions and possession-change confirmation |
| Dead-Ball Logic | Pitch-boundary checks, tracking-based stoppage detection and Smart Tagging-supported set pieces |
| State and Streaming | Possession state management, missing-ball handling, chunk processing and reproducibility |
| Integration and Validation | Pipeline execution, CSV generation, automated tests and submission-format validation |

The main implementation is distributed across `synx/data.py`, `synx/possession_features.py`, `synx/ball_possession.py`, `synx/approaches.py`, `synx/possession.py`, `synx/pitch.py`, `synx/stoppages.py`, `synx/spells.py` and `synx/streaming.py`.

## 14. Output Generation

After processing, the system generates a CSV file named `Synx.csv`.

The output contains two columns:

| Column | Description |
|---|---|
| `frame` | Frame number |
| `label` | Predicted possession label |

Example:

```csv
frame,label
0,X
1,H
2,H
3,H
4,A
5,D
```

The output contains one prediction for every frame of the match.

## 15. Installation and Execution

### Requirements

The project uses Python and the dependencies listed in `requirements.txt`.

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the required tactical tracking files and Smart Tagging file in the expected data directory, or configure `MTAH_DATA_DIR` to point to the data location.

### Run Prediction

```bash
python run_all.py --output Synx.csv --overwrite
```

The default run uses `tagged` dead-ball detection.

To select another mode:

```bash
python run_all.py --dead-ball basic --output Synx.csv --overwrite
python run_all.py --dead-ball tracking --output Synx.csv --overwrite
python run_all.py --dead-ball tagged --output Synx.csv --overwrite
```

### Validate Output

```bash
python evaluation/validate_submission.py Synx.csv
```

The final generated file is `Synx.csv`, containing the predicted possession label for every frame in the required `frame,label` format.

---

**Team Synx**  
*More Than A Hack 2027*
