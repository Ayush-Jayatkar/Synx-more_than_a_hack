# Synx - More Than a Hack 2027

Frame-by-frame football possession labels from player and ball tracking data. Team Synx.

## 1. Overview

For every frame of the tactical video (BAR-ATH, 199,319 frames at 25 FPS) the pipeline writes one label:

| Label | Meaning |
|---|---|
| H | Home team has possession |
| A | Away team has possession |
| D | Dead ball (match not in play, or ball clearly outside the pitch) |
| X | Abstain (no established possession, or the ball has been unseen too long) |

The method is rule-based. It has no machine learning, no training and no fitted parameters. It uses the
tracking data (ATD) only. The technical report is in [Synx.md](Synx.md).

## 2. Architecture

```text
data/tactical/*_atd.xml + *_atd.txt
        |
        v
synx/data.py             parse ATD, convert to metres, cache (cache/BAR-ATH_tracking.npz)
        |
        v
synx/possession.py       dead-ball mask D (synx/pitch.py: half windows, ball beyond pitch lines)
synx/stoppages.py        optional extra D evidence, chosen with --dead-ball (command-line default tagged)
synx/possession_features.py   nearest-player distances, ball motion, approach speed
        |
        v
synx/approaches.py       E_combined frame-level control evidence (synx/ball_possession.py)
        |
        v
synx/spells.py           possession state machine: hysteresis, backdated switch, missing-ball grace
synx/streaming.py        runs the above in chunks, same labels for any chunk size
        |
        v
run_all.py  ->  outputs/workstream2/Synx.csv   (frame,label)
```

`run_all.py` calls `streaming.predict_chunked`. `synx.possession.predict_possession` is the same engine as a
library call and gives the same labels.

Options:

- `--dead-ball` chooses the stoppage evidence added to D (section 8). The command-line default is `tagged`, which writes the final submission. `--dead-ball basic` reproduces the previous submission.
- `--smart-tagging` also writes a diagnostics CSV (`frame_diagnostics.csv`) next to the output. It does not change labels.
- `interfaces.py` and `modules/` are offline helpers (module contract, probability adapter, nearest-player
  baseline). `run_all.py` does not use them. From `starter/`, the pipeline uses only `team_of_each_track`
  (`starter/baseline.py`, slot-to-team mapping) and `KICKOFF_VIDEO_FRAMES` (`starter/load_smart_tagging.py`); the
  nearest-player baseline itself is not run.
- `synx/missing_ball.py` and `synx/tracking_diagnostics.py` are not used by `run_all.py`.

## 3. Folder structure

Tracked files only. `data/`, `cache/` and `outputs/` are created locally and are gitignored.

```text
.
├── run_all.py               entry point: writes Synx.csv
├── games.py                 per-game constants, data folder lookup (MTAH_DATA_DIR)
├── synx_validate.py         wrapper around the official format validator
├── verify_synx.py           technical checks on Synx.csv + diagnostics CSV
├── interfaces.py            optional module contract (not used by run_all.py)
├── requirements.txt
├── Synx.md                  technical report
├── synx/
│   ├── data.py              ATD parsing, metres, cache
│   ├── pitch.py             in-play window, ball-outside-pitch test
│   ├── possession.py        dead mask and library API
│   ├── possession_features.py   distances, motion, approach speed
│   ├── approaches.py        control evidence models (E_combined is used)
│   ├── ball_possession.py   E_combined score formula
│   ├── spells.py            possession state machine
│   ├── streaming.py         chunked engine
│   ├── events.py            optional Smart Tagging alignment (diagnostics only)
│   ├── diagnostics.py       optional diagnostics CSV writer
│   ├── missing_ball.py      not used by run_all.py
│   └── tracking_diagnostics.py  not used by run_all.py
├── modules/                 optional offline adapters (not used by run_all.py)
├── starter/                 organizer starter code (load_atd.py, baseline.py, notebooks, ...)
├── evaluation/              organizer files: validate_submission.py, possession_metrics.py,
│                            run_evaluation.py, results/ (baseline_*.csv only)
├── tests/                   pytest suites (synthetic data only; no match data needed)
└── docs/                    WORKSTREAM1.md, WORKSTREAM2.md, PROBABILITY_INTEGRATION.md
```

## 4. Python and dependencies

- Python 3.12 (as stated in `requirements.txt`). The 2026-11-17 validation run used Python 3.13.7.
- `numpy==2.1.3`, `pandas==2.2.3` (pinned in `requirements.txt`).
- `pytest` is needed only to run the tests. It is not in `requirements.txt`; install it with `pip install pytest`.

## 5. Environment setup

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 6. Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest        # only for the tests
```

## 7. Data path

The match data is not in this repository. Download the official BAR-ATH bundle from the More Than a Hack 2027 organisers (Metrica Sports) and put it in `./data`, or point `MTAH_DATA_DIR` at it
(`games.py` reads this variable; the default is `./data` next to `games.py`).

```bash
export MTAH_DATA_DIR=/path/to/mtah2027          # Linux / macOS
```
```powershell
$env:MTAH_DATA_DIR = "D:\mtah2027"              # Windows PowerShell
```

The default run (`--dead-ball tagged`) reads these three files:

```text
<data dir>/tactical/BAR-ATH_tactical_atd.xml
<data dir>/tactical/BAR-ATH_tactical_atd.txt
<data dir>/tactical/BAR-ATH_tactical_smart_tagging.xml
```

`--dead-ball basic` or `--dead-ball tracking` needs only the two ATD files. `--smart-tagging` also reads the Smart Tagging XML.
Homography, video and the `broadcast/` folder are not read.

The first run parses the ATD and writes `cache/BAR-ATH_tracking.npz`. The cache is rebuilt automatically when the
source file size or modification time changes, or when `CACHE_VERSION` in `synx/data.py` changes.

## 8. Run

```bash
python run_all.py
```

Writes `outputs/workstream2/Synx.csv` (the tagged submission, SHA-256 `67c29b1642fd7b5e348160502679cd66b397e999ceeeeb286c06a5e02b9055a0`). An existing output is not replaced unless `--overwrite` is given.

Options (from `run_all.py`):

| Option | Default | Effect |
|---|---|---|
| `--output PATH` | `outputs/workstream2/Synx.csv` | exact path of the submission CSV |
| `--overwrite` | off | allow replacing an existing output / diagnostics file |
| `--chunk-size N` | 4096 | frames per streaming chunk (does not change the labels) |
| `--smart-tagging` | off | also write `frame_diagnostics.csv` in the output folder |
| `--dead-ball MODE` | `tagged` | `basic`, `tracking` or `tagged` (see below) |

`--dead-ball` chooses the dead-ball (`D`) evidence:

- `basic`: half windows and ball beyond the pitch lines. Output is byte-identical to the previous submission, kept as a
  fallback (SHA-256 `c46a268a6bed3024405498d236b922e67a0121b66d2d5f006e16c1d77c0285fb`).
- `tracking`: `basic` plus tracking-only evidence. Still ball: the ball is seen with filtered speed below 0.5 m/s for
  at least 2 s. Out-of-play bridge: unseen frames right after a frame where the ball was seen out, for at most 30 s.
- `tagged` (default, final submission): `tracking` plus Smart Tagging set pieces. It reads `data/tactical/BAR-ATH_tactical_smart_tagging.xml`.

Only the command-line default changed: library calls (`run_all.predict`, `run_all.run`, `possession.predict_possession`) with no dead mask still use the basic mask. A missing ball alone never becomes `D`. The mask is built on the
whole match before chunked inference, so labels do not depend on `--chunk-size`. The rules are in `synx/stoppages.py`
and Synx.md section 11.

## 9. Regenerate Synx.csv

```bash
python run_all.py --output Synx.csv --overwrite
```

Other examples:

```bash
python run_all.py --chunk-size 1024
python run_all.py --smart-tagging --overwrite   # also writes outputs/workstream2/frame_diagnostics.csv
python run_all.py --dead-ball basic --overwrite   # previous submission, needs only the two ATD files
python run_all.py --dead-ball tracking --overwrite
python run_all.py --dead-ball tagged --overwrite   # same as the default; needs the Smart Tagging XML locally
```

`Synx.csv` in the repository root is gitignored, like everything under `outputs/`.

## 10. Tests and validation

Synthetic tests (no match data needed). The whole suite under pytest:

```bash
python -m pytest tests/ -v
```

Each file also runs on its own as a script (`tests/test_module_architecture.py` runs under pytest only):

```bash
python -B tests/test_data.py
python -B tests/test_pipeline.py
python -B tests/test_possession_features.py
python -B tests/test_probability_module.py
python -B tests/test_streaming.py
python -B tests/test_stoppages.py
git diff --check
```

| File | What it covers |
|---|---|
| `test_data.py` | ATD parsing, malformed lines, NaN for missing observations, ball speed, cache |
| `test_possession_features.py` | nearest / second-nearest distances, speeds, approach speed, discontinuity filter |
| `test_pipeline.py` | control evidence, state machine, hysteresis, dead ball, missing-ball grace |
| `test_streaming.py` | chunked engine: identical labels for any chunk size, state carried across chunk boundaries, runner CLI |
| `test_stoppages.py` | still ball, out-of-play bridge, Smart Tagging set-piece windows and restarts, malformed events, mode handling (synthetic) |
| `test_probability_module.py`, `test_module_architecture.py` | optional offline helpers in `interfaces.py` and `modules/` |

`test_long_match_chunk_sizes_match_offline_engine` in `test_streaming.py` runs a 5,000-frame synthetic match at
chunk sizes 1, 25, 100, 512 and 4096. It checks every size against the whole-track run and
`possession.predict_possession`. The match includes a short ball gap across a chunk boundary (the label is held), a
ball gap longer than 1 s (the label becomes `X`) and dead runs across chunk boundaries (`D`).

Validation of a generated CSV (needs the match data, see section 7):

```bash
python evaluation/validate_submission.py outputs/workstream2/Synx.csv
python synx_validate.py outputs/workstream2/Synx.csv
python verify_synx.py --input outputs/workstream2/Synx.csv --diagnostics outputs/workstream2/frame_diagnostics.csv
```

Reproducibility checks (labels must not depend on chunk size; two runs must be byte-identical). These use the default, so they produce the tagged output:

```bash
python run_all.py --chunk-size 256  --output outputs/chunk256.csv  --overwrite
python run_all.py --chunk-size 4096 --output outputs/chunk4096.csv --overwrite
python run_all.py --output outputs/hash_run1.csv --overwrite
python run_all.py --output outputs/hash_run2.csv --overwrite
```
```powershell
(Get-FileHash .\outputs\chunk256.csv).Hash -eq (Get-FileHash .\outputs\chunk4096.csv).Hash
(Get-FileHash .\outputs\hash_run1.csv).Hash -eq (Get-FileHash .\outputs\hash_run2.csv).Hash
```

Both lines print `True`. The default output hash is `67c29b1642fd7b5e348160502679cd66b397e999ceeeeb286c06a5e02b9055a0`.

Structural check of the CSV (expect 199319 rows, columns `['frame', 'label']`, labels H/A/D/X only, 0 duplicates,
0 nulls, frames 0 to 199318):

```bash
python -c "import pandas as pd; d=pd.read_csv('outputs/workstream2/Synx.csv'); print(len(d), list(d.columns), d.label.value_counts().to_dict(), d.frame.duplicated().sum(), d.isnull().sum().sum(), d.frame.min(), d.frame.max())"
```

- The tests use synthetic arrays only. They do not need match data.
- `validate_submission.py` and `synx_validate.py` check the file format only (row count, frame numbers, labels). They use no ground truth.
- `verify_synx.py` requires `--diagnostics`. Create that file first with the `--smart-tagging` command in
  section 9. The script checks that the two files have the same frames. It writes `verification_report.json`
  next to the `--input` file.

What these checks show:

- The synthetic tests check that the logic is correct on constructed inputs.
- The validators and the structural check confirm the output format.
- The chunk and hash checks confirm the output is reproducible.
- None of them measures prediction accuracy. `evaluation/run_evaluation.py` and the ground truth are not part of
  this workflow. No parameter was chosen from a score. The only accuracy numbers are the one-time public sample
  check below.

Last local run (2026-11-17, Python 3.13.7, numpy 2.1.3, pandas 2.2.3):

- 130 tests passed.
- The official validator reported OK.
- `verify_synx.py` completed.
- Chunk sizes 256, 1024, 4096 and the whole track gave identical labels, and two fresh runs gave identical files.
- `python run_all.py --dead-ball basic` SHA-256 (previous submission, kept as fallback): `c46a268a6bed3024405498d236b922e67a0121b66d2d5f006e16c1d77c0285fb`.

Real-data run of the default `--dead-ball tagged` (full match, no ground truth):

- Dead frames 46,528 (`basic`) to 57,928. Tracking rules added 8,551 (still ball 2,172 frames in 22 runs;
  out-of-play bridge 6,379). Smart Tagging added 2,849 more.
- Of 57 set pieces considered (the 2 `Kick Off (Start)` tags are excluded), 35 were located with a restart, 1 of 4
  after-goal kick offs was used and 21 were skipped for lack of stoppage evidence.
- Label changes against `basic`: 12,174 frames (X to D 9,192; H to D 1,297; A to D 911; H to X 325; A to X 449).
- Official validator OK, `verify_synx.py` OK, chunk sizes 256, 1024, 4096 and the whole track gave 0 mismatches, and
  two fresh runs gave the same SHA-256: `67c29b1642fd7b5e348160502679cd66b397e999ceeeeb286c06a5e02b9055a0`.

Public five-minute sample check, run once per version with `evaluation/run_evaluation.py --gt public`. It was never
used to tune anything, and the thresholds were fixed before it was run:

| Version | macroF1 | D F1 | spellF1 | transition F1 (+/-25 frames) |
|---|---|---|---|---|
| `basic` | 0.5511 | 0.125 | 0.4588 | 0.158 |
| `tagged` | 0.6697 | 0.481 | 0.4588 | 0.158 |

spellF1 and transition F1 did not move because the scorer forward-fills predicted `X` inside graded spans and treats
`X` and `D` alike for alive/dead. `X` frames that followed an existing `D` were already scored as `D` for spells and
transitions, and the new `D` mostly replaced such `X`. Five minutes (17 possessions) is not representative of the
full match. There is no full-match accuracy claim.

## 11. Output format

`Synx.csv` is a CSV with header `frame,label` and 199,319 rows. `frame` runs 0 to 199,318 and is the tactical
video frame index. `label` is one of `H`, `A`, `D`, `X` (see section 1). Frame `i` uses ATD `IdFrame` `i + 1`.

## Competition rules (short)

- Match data is input only. No training, fitting or tuning on ground truth or video.
- Do not compare output with the video and do not hand-label frames.
- Do not paste or upload match data, screenshots or extracts into AI tools or hosted services.
- Smart Tagging is an allowed input: the official starter kit (`starter/load_smart_tagging.py`,
  `starter/03_first_submission.ipynb`) recommends set pieces as dead-ball evidence, and the team confirmed it.
  The default `--dead-ball tagged` uses it. The XML is match data and is never committed.
- `data/`, `cache/`, `outputs/`, `evaluation/gt/`, `*.npz`/`*.pkl` caches, videos, `.atd` and `.xml` files stay out of Git.
- Ground truth (`evaluation/gt/`) is kept locally only. The pipeline never reads it.

## Team git workflow (short)

One branch per person, merged by pull request. Do not edit another person's module without telling them.

```bash
git checkout -b your-name/topic
git add <files>
git commit -m "Describe your change"
git push origin your-name/topic      # then open a PR
```

Repository: https://github.com/Ayush-Jayatkar/Synx-more_than_a_hack
