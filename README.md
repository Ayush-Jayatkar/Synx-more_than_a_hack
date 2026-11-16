# Synx-more_than_a_hack

**Football Possession Prediction Using Tracking Data and Rule-Based Temporal Smoothing**

## About the Project

Synx is a football analytics project focused on developing an intelligent system for frame-by-frame possession prediction.

The system aims to identify which team controls the ball throughout a match using player tracking information, ball movement and temporal patterns.

### Core Objectives

- Predict team possession at every frame.
- Identify possession changes and dead-ball situations.
- Handle missing and noisy tracking observations.
- Apply rule-based temporal smoothing (hysteresis, minimum spell length) for consistent predictions.
- Develop a reliable and reproducible prediction pipeline.

### Prediction Classes

| Label | Meaning |
|---|---|
| H | Home team possession |
| A | Away team possession |
| D | Dead ball |
| X | Abstain |

### Technology Stack

- Python
- NumPy and Pandas
- Git and GitHub

No machine learning: the final pipeline combines tracking-derived evidence with a temporal possession state machine.

## Architecture

`run_all.py` is the final submission-generation pipeline. It loads tracking data, derives the default dead-ball mask, and feeds the Workstream 2 features and possession evidence into the chunked streaming state machine. That engine handles possession switches, missing-ball abstention, and spell continuity, then writes one `H`, `A`, `D`, or `X` label per frame in the official `frame,label` format. The default output is `outputs/workstream2/Synx.csv`; chunk size is configurable, and existing outputs are protected unless `--overwrite` is supplied. Optional Smart Tagging context produces a local diagnostics sidecar and does not change predictions.

Validate a generated CSV separately with `python synx_validate.py <submission.csv>`. This wrapper calls the official format validator in `evaluation/validate_submission.py`; validation does not use ground truth. `modules/synx_combined.py` is a separately tested H/A/D probability adapter for the advanced evidence and is not used by the final submission pipeline. There is no shared plug-in registry or probability blender in the final path.

`starter/baseline.py` is an optional nearest-player reference baseline, not the final prediction algorithm. Its standalone outputs are for comparison only and are not used by `run_all.py`.

## Project Structure

```text
Synx-more_than_a_hack/
│
├── evaluation/
│   ├── results/
│   ├── possession_metrics.py
│   ├── run_evaluation.py
│   └── validate_submission.py
│
├── starter/
│   └── baseline.py          optional nearest-player reference
│
├── synx/
│   ├── data.py              tracking input and cache
│   ├── pitch.py             pitch geometry and in-play windows
│   ├── possession_features.py distance, movement, and spatial features
│   ├── approaches.py        possession evidence
│   ├── possession.py        prediction API and dead-ball handling
│   ├── spells.py            temporal state machine and spell handling
│   ├── streaming.py         stateful chunk processing
│   ├── events.py            optional event-context alignment
│   └── diagnostics.py       local diagnostic reports
│
├── modules/synx_combined.py H/A/D probability adapter
├── tests/                    synthetic feature, probability, and streaming checks
├── run_all.py               inference, diagnostics and format validation
├── Synx.md                  technical report (upload to the form)
├── games.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Getting Started


### 1. Clone the Repository

```bash
git clone https://github.com/Ayush-Jayatkar/Synx-more_than_a_hack.git
```

### 2. Navigate to the Project

```bash
cd Synx-more_than_a_hack
```

### 3. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Point to the data (not in this repo)

```bash
export MTAH_DATA_DIR=/path/to/data      # macOS / Linux
$env:MTAH_DATA_DIR = "D:\\data"         # Windows PowerShell
```

### 6. Reproduce Synx.csv

```bash
python run_all.py --chunk-size 4096 --smart-tagging
```

This writes `outputs/workstream2/Synx.csv` (`frame,label`) and, with `--smart-tagging`,
a frame-level diagnostics sidecar under `outputs/workstream2/`. Validate the CSV separately
with `python synx_validate.py outputs/workstream2/Synx.csv`. Use `--overwrite` only when
intentionally replacing an existing output. Local outputs and caches are ignored by Git.

The first run parses the tracking file and caches it in `cache/` (gitignored); later runs are fast.

Self-checks on synthetic data (no match data needed):

```bash
python tests/test_data.py
python tests/test_pipeline.py
python tests/test_possession_features.py
python tests/test_probability_module.py
python tests/test_streaming.py
```

### 7. Open in VS Code

```bash
code .
```

## Git Workflow

One branch per person, merged by pull request. Don't edit another person's module without telling them.

```bash
git checkout main && git pull origin main
git checkout -b your-name/module
git add synx/your_module.py
git commit -m "Describe your changes"
git push origin your-name/module      # then open a PR
```

## Competition Rules (read before coding)

- Match data is **input only**. Never train, fit or tune on the ground truth or the video.
- Never compare our output with the video, and never hand-label frames.
- The public 5-min ground truth is only for checking the pipeline runs.
- Never paste or upload match data, screenshots or extracts into AI tools.
- `data/`, videos, `.atd` and `.xml` files are git-ignored. Keep it that way.

---

**Team Synx**
