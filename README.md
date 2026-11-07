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

No machine learning: the pipeline is a fixed rule-based algorithm. Every parameter is justified in `REPORT.md`.

## Project Structure

```text
Synx-more_than_a_hack/
│
├── evaluation/
│   ├── gt/
│   ├── results/
│   ├── possession_metrics.py
│   ├── run_evaluation.py
│   └── validate_submission.py
│
├── starter/
│   ├── 01_smart_tagging.ipynb
│   ├── 02_tracking_atd.ipynb
│   ├── 03_first_submission.ipynb
│   ├── baseline.py
│   ├── load_atd.py
│   ├── load_homography.py
│   ├── load_smart_tagging.py
│   ├── make_dummy_submission.py
│   └── utils.py
│
├── synx/
│   ├── data.py              (A) tracking -> per-frame table
│   ├── pitch.py             (B) homography, pitch coords, Smart Tagging
│   ├── ball_possession.py   (C) nearest player, distances, control
│   ├── missing_ball.py      (D) gap filling, state carry, confidence
│   └── spells.py            (E) dead ball, hysteresis, spells, X policy
│
├── run_all.py               writes Synx.csv and validates it
├── REPORT.md
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
python run_all.py
```

This writes `Synx.csv` (199,319 rows, `frame,label`) and runs `evaluation/validate_submission.py` on it.

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

