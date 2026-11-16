"""Run chunked possession inference and write an official-format submission."""
import argparse
import os

import numpy as np
import pandas as pd

from synx import approaches, data, diagnostics, events, possession, possession_features, streaming

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(ROOT, "outputs", "workstream2", "Synx.csv")


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    """Create the command-line parser without loading any local match assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=_positive_int, default=4096,
                        help="tracking frames processed per streaming chunk")
    parser.add_argument("--smart-tagging", action="store_true",
                        help="add optional Smart Tagging context to a local diagnostic report")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="exact path for the official frame,label submission CSV")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing submission or diagnostic sidecar")
    return parser


def predict(track, chunk_size=4096, dead=None):
    """Predict through the shared chunked streaming implementation."""
    dead = possession.default_dead_mask(track) if dead is None else np.asarray(dead, dtype=bool)
    return streaming.predict_chunked(track, dead, chunk_size=chunk_size)


def write_submission(path, labels):
    """Write the required frame/label CSV from an already predicted label vector."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, ("H", "A", "D", "X")).all():
        raise ValueError("labels must be a one-dimensional H/A/D/X array")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)


def _assert_writable(paths, overwrite):
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): {}. Use --overwrite to replace them.".format(
                ", ".join(existing)
            )
        )


def _write_event_diagnostics(path, track, labels, dead, event_context):
    """Write local context only; events do not affect control or output labels."""
    features = possession_features.build_features(track)
    evidence = approaches.combined(features)
    diagnostics.write_frame_report(
        path, labels, features, evidence["control"], dead,
        confidence=evidence["confidence"], event_context=event_context,
    )


def run(track, output, chunk_size=4096, overwrite=False, event_context=None):
    """Run inference and write output, optionally adding an event-context sidecar."""
    output = os.path.abspath(output)
    diagnostic_path = os.path.join(os.path.dirname(output), "frame_diagnostics.csv")
    paths = [output] + ([diagnostic_path] if event_context is not None else [])
    _assert_writable(paths, overwrite)

    dead = possession.default_dead_mask(track)
    labels = predict(track, chunk_size=chunk_size, dead=dead)
    write_submission(output, labels)
    if event_context is not None:
        _write_event_diagnostics(diagnostic_path, track, labels, dead, event_context)
    return labels


def main(argv=None):
    """Parse CLI arguments and run local tracking inference."""
    args = build_parser().parse_args(argv)
    track = data.load()
    event_context = None
    if args.smart_tagging:
        raw_events = events.load_smart_tagging_events()
        event_context = events.align_events(raw_events, len(track["ball"]))
    labels = run(track, args.output, args.chunk_size, args.overwrite, event_context)
    print("Submission saved: {}".format(os.path.abspath(args.output)))
    if event_context is not None:
        print("Diagnostics saved: {}".format(
            os.path.join(os.path.dirname(os.path.abspath(args.output)), "frame_diagnostics.csv")
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
