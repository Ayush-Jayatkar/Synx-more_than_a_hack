# Team Synx – Technical Documentation
## More Than a Hack 2027 | Football Possession Prediction

## 1. Project Overview

Synx is a football possession prediction system that uses player and ball tracking data to assign a possession label to every frame of a match.

It analyses player positions, distances, movement, team support and opponent pressure to estimate which team controls the ball. It also identifies dead-ball situations and maintains possession states across consecutive frames.

The system processes the BAR-ATH tactical match, containing 199,319 frames at 25 frames per second.

The output labels are:

| Label | Meaning |
|---|---|
| `H` | Home team possession |
| `A` | Away team possession |
| `D` | Dead-ball situation |
| `X` | Unknown or uncertain possession |

The final output is a CSV file with the columns `frame,label`, containing one row for every frame.

## 2. System Architecture

The prediction pipeline follows these stages:

Tracking Data → Data Preprocessing → Feature Extraction → Possession Estimation → Dead-Ball Detection → State Management → CSV Output

### Main Technical Modules

| Module | Responsibility |
|---|---|
| `synx/data.py` | Parses tracking data and creates player, ball and team arrays |
| `synx/possession_features.py` | Calculates distances, speeds and movement features |
| `synx/ball_possession.py` | Calculates possession scores and frame-level control |
| `synx/approaches.py` | Implements possession approaches and selects E_combined |
| `synx/possession.py` | Provides possession prediction and basic dead-ball detection |
| `synx/pitch.py` | Defines pitch dimensions and boundary checks |
| `synx/stoppages.py` | Detects tracking-based and Smart Tagging-supported stoppages |
| `synx/spells.py` | Maintains possession states and confirms possession changes |
| `synx/streaming.py` | Processes frames in chunks while preserving state |
| `run_all.py` | Executes the complete pipeline and generates the CSV |

## 3. Data Loading and Preprocessing

The system reads tactical tracking data containing player and ball positions and speeds.

The parser converts the input into structured arrays:

- `ball`: Ball coordinates for each frame.
- `players`: Coordinates of tracked player slots.
- `player_speed`: Player speed in metres per second.
- `team`: Mapping of each player slot to the home or away team.
- `ball_speed`: Speed calculated from consecutive ball positions.
- `pitch`: Pitch dimensions used for coordinate conversion.

The match contains 46 player tracking slots: 24 home and 22 away. These represent tracking identities and may appear or disappear during the match.

### Coordinate Conversion

Raw tracking coordinates are normalised between 0 and 1 and converted into metres using the pitch dimensions:

- Pitch length: 105.3 metres
- Pitch width: 68.0 metres

Tracking frame identifiers start at 1, while video frames start at 0. The parser converts them into matching zero-based frame indices.

Missing coordinates are represented as `NaN`.

### Tracking Cache

Parsed arrays are stored locally in:

`cache/BAR-ATH_tracking.npz`

The cache is rebuilt when the source-file information or cache version changes.

## 4. Feature Extraction

Feature extraction is implemented in `synx/possession_features.py`.

For each frame, the system calculates:

- **Nearest-player distance:** Distance between the ball and the closest player of each team.
- **Second-nearest distance:** Distance to the second-closest player.
- **Nearest-player speed:** Speed of the closest tracked player.
- **Players nearby:** Number of teammates within the 2.0-metre control radius.
- **Distance advantage:** Difference between home and away nearest-player distances.
- **Approach speed:** Rate at which the nearest player is moving closer to the ball.
- **Ball speed and direction:** Calculated using consecutive ball positions.
- **Ball visibility:** Indicates whether valid ball coordinates are available.

Approach speed is checked for tracking discontinuities. Sudden changes that cannot be explained by player and ball movement are rejected to prevent incorrect possession evidence.

## 5. Possession Prediction – E_combined

The production pipeline uses the **E_combined approach** to estimate which team controls the ball in each frame.

A score is calculated independently for both teams. A lower score represents stronger evidence of control.

The calculation considers:

1. Distance between the nearest player and the ball.
2. Support from nearby teammates.
3. Pressure from nearby opponents.
4. Player approach speed towards the ball.
5. Difference between ball speed and nearest-player speed.

A player close to the ball provides evidence of control. Nearby teammates provide support, while nearby opponents increase pressure. A player moving towards the ball receives a small contribution, while a ball moving much faster than the nearest player reduces the likelihood of control.

### Control Decision

A team is assigned control when:

- Its nearest player is within the 2.0-metre control radius.
- Its possession score is better than the opposing team's score by the required margin.

If neither team satisfies these conditions, the frame is treated as having no confirmed control.

Other approaches are available in `synx/approaches.py`, but the production pipeline uses E_combined.

## 6. Possession State Management

Frame-level control estimates can fluctuate because of tackles, deflections, loose balls and tracking noise.

The system uses a state machine implemented in `synx/spells.py` to maintain consistent possession labels.

### Possession Confirmation

A new team must show control for 13 consecutive frames before a possession change is confirmed.

At 25 FPS, this represents approximately 0.52 seconds.

When a change is confirmed, buffered candidate frames are assigned to the new team from the first candidate frame. If the candidate is interrupted, the previous possession state is retained.

The system also allows one unseen frame during a pending possession change.

### Contested and Loose Balls

When both teams have players close to the ball, their possession scores are compared. If neither team has sufficiently strong evidence, no new control is assigned.

When no player is within the control radius, the ball is treated as loose. The previous possession label is retained until a confirmed state transition occurs.

## 7. Dead-Ball Detection

Dead-ball detection identifies periods when normal gameplay is interrupted.

The system supports three modes:

| Mode | Detection Method |
|---|---|
| `basic` | Match in-play windows and ball-out-of-pitch detection |
| `tracking` | Basic detection plus still-ball and out-of-play tracking evidence |
| `tagged` | Tracking detection with Smart Tagging set-piece events |

The default command-line mode is `tagged`.

### Basic Detection

A frame is marked as dead when it falls outside the defined in-play period or when the visible ball is more than 0.5 metres beyond the pitch boundary.

### Tracking-Based Detection

The tracking mode adds:

- **Still ball:** Detects a visible ball moving below 0.5 m/s for at least 50 frames (2 seconds).
- **Out-of-play bridge:** Extends an out-of-play period across missing-ball frames for up to 30 seconds.

### Smart Tagging-Supported Detection

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

## 12. Limitations

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

## 14. Installation and Execution

### Requirements

The project uses Python and the dependencies listed in `requirements.txt`.

Create a virtual environment:

```bash
python -m venv .venv
