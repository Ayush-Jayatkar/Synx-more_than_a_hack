"""
run_all.py - tracking data -> validated submission CSV, one command.

    python run_all.py                                   # baseline, validate + score
    python run_all.py --modules nearest_player,hmm --weights 1,2 --name blend1
    python run_all.py --window 75 --min-spell 100 --abstain 0.4 --name tuned

Output: evaluation/results/<name>.csv  (columns: frame,label ; 199319 rows).
Only the graded blocks (games.half_video_blocks) get H/A/D; every other frame is X.
Run from the repo root.
"""
import argparse
import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from interfaces import LABELS, check_output, normalize  # noqa: E402

# name -> "module_file:ClassName". Each member adds ONE line.
MODULE_REGISTRY = {
    "nearest_player": "modules.nearest_player:NearestPlayer",
    # "hmm":  "modules.hmm_module:HMMModule",
    "nearest_player_long": "modules.nearest_player:NearestPlayerLong",
}

# Central knobs (frames at 25 fps). Tune with run_evaluation.py, not inside modules.
WINDOW = 51        # moving-average window on probabilities (~2 s)
MIN_SPELL = 75     # possession runs shorter than this are merged into a neighbour (~3 s)
ABSTAIN = 0.0      # label X where smoothed confidence < this (0 = never abstain)


def load_module(name):
    path, cls = MODULE_REGISTRY[name].split(":")
    return getattr(importlib.import_module(path), cls)()


def blend(probs, weights):
    w = np.asarray(weights, dtype=np.float64)
    return normalize(sum(wi * p for wi, p in zip(w / w.sum(), probs)))


def moving_average(p, window):
    if window <= 1:
        return p
    k = np.ones(window) / window
    return np.stack([np.convolve(p[:, c], k, mode="same") for c in range(p.shape[1])], axis=1)


def _runs(x):
    change = np.flatnonzero(np.diff(x) != 0) + 1
    return np.r_[0, change], np.r_[change, len(x)]


def merge_short(idx, min_len):
    """Merge runs shorter than min_len into the longer neighbouring run, shortest first."""
    idx = idx.copy()
    while True:
        starts, ends = _runs(idx)
        lens = ends - starts
        short = np.flatnonzero(lens < min_len)
        if len(short) == 0 or len(starts) == 1:
            return idx
        k = short[np.argmin(lens[short])]
        left, right = (k - 1 if k > 0 else None), (k + 1 if k < len(starts) - 1 else None)
        if left is None:
            tgt = right
        elif right is None:
            tgt = left
        else:
            tgt = left if lens[left] >= lens[right] else right
        idx[starts[k]:ends[k]] = idx[starts[tgt]]


def make_labels(prob, blocks, n_frames, window=WINDOW, min_spell=MIN_SPELL, abstain=ABSTAIN):
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, a, b in blocks:
        a, b = max(a, 0), min(b, n_frames - 1)
        sm = moving_average(prob[a:b + 1], window)
        idx = merge_short(sm.argmax(axis=1), min_spell)
        lab = LABELS[idx]
        lab[sm.max(axis=1) < abstain] = "X"
        labels[a:b + 1] = lab
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="nearest_player")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--name", default="submission")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--min-spell", type=int, default=MIN_SPELL)
    ap.add_argument("--abstain", type=float, default=ABSTAIN)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    from games import GAME, GAMES, half_video_blocks
    from synx.data import load

    names = a.modules.split(",")
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(names)
    if len(weights) != len(names):
        sys.exit("--weights must have one value per module")

    n_frames = GAMES[GAME]["n_video_frames"]
    track = load()
    if len(track["ball"]) != n_frames:
        sys.exit("track has {} frames, expected {}".format(len(track["ball"]), n_frames))

    probs = []
    for nm in names:
        print("running", nm)
        probs.append(check_output(load_module(nm).predict_proba(track), n_frames, nm))
    prob = blend(probs, weights)

    labels = make_labels(prob, half_video_blocks(GAME), n_frames,
                         a.window, a.min_spell, a.abstain)
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, a.name + ".csv")
    pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels}).to_csv(out, index=False)
    vals, counts = np.unique(labels, return_counts=True)
    print("wrote", out, dict(zip(vals, counts.tolist())))

    if not a.no_validate:
        if subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out]):
            sys.exit("VALIDATION FAILED - do not submit this file")
    if not a.no_eval:
        subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")])


if __name__ == "__main__":
    main()