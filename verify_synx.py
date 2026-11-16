
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Synx technical verification")
    parser.add_argument("--input", required=True, help="Synx.csv path")
    parser.add_argument("--diagnostics", required=True, help="frame_diagnostics.csv path")
    parser.add_argument("--verbose", action="store_true",
                        help="print the detailed analysis tables")
    args = parser.parse_args()
    detail = print if args.verbose else (lambda *values, **kwargs: None)

    detail("Synx Technical Verification")

    pred = pd.read_csv(args.input)
    diag = pd.read_csv(args.diagnostics)

    required_pred = {"frame", "label"}
    required_diag = {
        "frame", "label", "confidence_m", "ball_seen", "dead",
        "contested", "loose", "possible_turnover"
    }

    if not required_pred.issubset(pred.columns):
        raise ValueError("Submission CSV is missing required columns")
    if not required_diag.issubset(diag.columns):
        raise ValueError("Diagnostics CSV is missing required columns")

    # 1. Submission integrity
    valid_labels = {"H", "A", "D", "X"}
    valid = pred["label"].isin(valid_labels).all()
    sequential = np.array_equal(pred["frame"].to_numpy(), np.arange(len(pred)))
    unique = pred["frame"].is_unique

    detail("\nSubmission Integrity")
    detail("Rows:", len(pred))
    detail("Valid labels:", valid)
    detail("Sequential frames:", sequential)
    detail("Unique frames:", unique)
    detail("Label distribution:")
    detail(pred["label"].value_counts().to_string())

    # Align diagnostics by frame
    if len(pred) != len(diag):
        raise ValueError("Prediction and diagnostic row counts differ")

    if not np.array_equal(pred["frame"], diag["frame"]):
        raise ValueError("Prediction and diagnostic frame alignment differs")

    df = diag.copy()
    df["missing_ball"] = ~df["ball_seen"].fillna(False).astype(bool)

    # 2. Missing ball
    detail("\nMissing-Ball Analysis")
    detail("Missing frames:", int(df["missing_ball"].sum()))
    detail("Missing percentage:",
          round(df["missing_ball"].mean() * 100, 2))

    detail("\nLabels during missing tracking:")
    detail(df.loc[df["missing_ball"], "label"].value_counts().to_string())

    detail("\nLabels during visible tracking:")
    detail(df.loc[~df["missing_ball"], "label"].value_counts().to_string())

    # Consecutive missing-ball runs
    changes = df["missing_ball"].ne(
        df["missing_ball"].shift(fill_value=False)
    ).cumsum()

    runs = df[df["missing_ball"]].groupby(changes).agg(
        start=("frame", "first"),
        end=("frame", "last"),
        frames=("frame", "size")
    )

    runs["seconds"] = runs["frames"] / 25

    detail("\nMissing-ball runs:", len(runs))
    if len(runs):
        detail("Median duration:", round(runs["seconds"].median(), 2), "s")
        detail("Longest duration:", round(runs["seconds"].max(), 2), "s")
        detail("Runs over 30 seconds:", int((runs["seconds"] > 30).sum()))

    # 3. Transitions
    detail("\nTransition Analysis")
    labels = df["label"].astype(str).to_numpy()
    valid_team = np.isin(labels, ["H", "A"])

    changes = np.where(
        (labels[1:] != labels[:-1]) &
        valid_team[1:] & valid_team[:-1]
    )[0] + 1

    transitions = df.iloc[changes].copy()
    detail("H/A transitions:", len(transitions))

    if len(changes) > 1:
        gaps = np.diff(changes) / 25
        detail("Median time between transitions:",
              round(float(np.median(gaps)), 2), "s")
        detail("Transitions within 2 seconds:",
              int((gaps < 2).sum()))

    if len(transitions):
        detail("Transitions with missing ball:",
              int(transitions["missing_ball"].sum()))
        detail("Transitions flagged possible turnover:",
              int(transitions["possible_turnover"].fillna(False).astype(bool).sum()))

    # 4. Confidence
    detail("\nConfidence Analysis")
    detail(df.groupby("label")["confidence_m"].agg(
        ["count", "mean", "median", "std"]
    ).round(3).to_string())

    # 5. Spell durations
    detail("\nPredicted Spell Durations")
    # Count contiguous runs of the same team label.
    team = pd.Series(labels).where(pd.Series(labels).isin(["H", "A"]))
    spell_ids = team.ne(team.shift()).cumsum()

    spells = pd.DataFrame({
        "team": team,
        "spell_id": spell_ids
    }).dropna().groupby(["spell_id", "team"]).size()

    durations = spells / 25

    if len(durations):
        detail("H/A spells:", len(durations))
        detail("Mean duration:", round(durations.mean(), 2), "s")
        detail("Median duration:", round(durations.median(), 2), "s")
        detail("Spells under 2 seconds:", int((durations < 2).sum()))

    # Save a compact report
    report = {
        "frames": len(pred),
        "format_valid": bool(valid and sequential and unique),
        "missing_ball_frames": int(df["missing_ball"].sum()),
        "missing_ball_runs": len(runs),
        "longest_missing_run_seconds": (
            round(float(runs["seconds"].max()), 2) if len(runs) else 0
        ),
        "ha_transitions": len(transitions),
        "ha_spells": len(durations),
        "confidence_by_label": {
            str(k): {a: (None if pd.isna(v) else float(v))
                     for a, v in row.items()}
            for k, row in df.groupby("label")["confidence_m"]
                .agg(["mean", "median"]).to_dict(orient="index").items()
        }
    }

    output = os.path.join(os.path.dirname(args.input),
                          "verification_report.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not args.verbose:
        print("Synx Technical Verification")
        print("Submission Integrity: {} rows; valid labels, sequential frames, unique frames: {}".format(
            len(pred), bool(valid and sequential and unique)
        ))
        print("Missing-Ball Analysis: {} frames ({:.2f}%); {} runs".format(
            int(df["missing_ball"].sum()), df["missing_ball"].mean() * 100, len(runs)
        ))
        print("Transition Analysis: {} H/A transitions".format(len(transitions)))
        print("Confidence Analysis: per-label details saved in the report")
        print("Predicted Spell Durations: {} H/A spells; mean {:.2f} s".format(
            len(durations), float(durations.mean()) if len(durations) else 0.0
        ))
    print("Technical verification completed.")
    print("Report saved:", output)


if __name__ == "__main__":
    main()
