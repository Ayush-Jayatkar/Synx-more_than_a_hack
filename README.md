# Synx-more_than_a_hack

**Football Possession Prediction Using Tracking Data and Temporal Modelling**

## About the Project

Synx is a football analytics project focused on developing an intelligent system for frame-by-frame possession prediction.

The system aims to identify which team controls the ball throughout a match using player tracking information, ball movement and temporal patterns.

### Core Objectives

- Predict team possession at every frame.
- Identify possession changes and dead-ball situations.
- Handle missing and noisy tracking observations.
- Apply temporal modelling to improve prediction consistency.
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
- SciPy
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

Additional libraries may be introduced as development progresses.

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

### 5. Open in VS Code

```bash
code .
```

## Git Workflow

**Pull latest changes:**

```bash
git pull origin main
```

**Save and push changes:**

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

---

**Team Synx**

