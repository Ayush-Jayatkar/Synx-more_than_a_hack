"""Validate a Synx submission with a concise terminal summary."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VALIDATOR = ROOT / "evaluation" / "validate_submission.py"
FAILURE_LOG = ROOT / "cache" / "synx_validation_error.log"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Synx frame,label CSV")
    parser.add_argument("--rows", type=int, default=None,
                        help="expected frame count (defaults to the official validator setting)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = [sys.executable, str(VALIDATOR), args.submission]
    if args.rows is not None:
        command.extend(("--rows", str(args.rows)))
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_LOG.write_text(
            "Command: {}\n\nstdout:\n{}\nstderr:\n{}".format(
                subprocess.list2cmdline(command), result.stdout, result.stderr
            ),
            encoding="utf-8",
        )
        print("Validation failed: {}".format(os.path.basename(args.submission)))
        print("Details saved: {}".format(FAILURE_LOG.relative_to(ROOT)))
        return result.returncode

    # Import the official metrics implementation only after official validation
    # succeeds; its spell definition and label ordering remain authoritative.
    evaluation_dir = str(ROOT / "evaluation")
    if evaluation_dir not in sys.path:
        sys.path.insert(0, evaluation_dir)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import possession_metrics as pm
    from games import GAME, GAMES
    from validate_submission import describe

    table = pm.read_label_csv(args.submission)
    codes = pm.label_codes(table)
    fps = GAMES[GAME]["fps"]
    n_spells, mean_duration = describe(codes, fps)
    name = os.path.basename(args.submission)

    print(name)
    print("{} rows, frame 0..{}".format(len(codes), len(codes) - 1))
    print("\nLabel counts")
    for index, label in enumerate(pm.LABELS):
        count = int((codes == index).sum())
        percentage = 100.0 * count / len(codes) if len(codes) else 0.0
        print("  {}    {:>5}   {:4.1f}%".format(label, count, percentage))
    print("\n{} possessions, mean duration {:.1f} s".format(
        n_spells, mean_duration
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
