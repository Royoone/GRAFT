"""Command-line entry point for one-context GRAFT prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .predict import GRAFTPredictor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GRAFT transcriptional response prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict query drugs in one target context")
    predict.add_argument("--bundle", type=Path, default=Path("resources"))
    predict.add_argument("--genetic", type=Path, required=True, help="Prepared genetic-response CSV")
    predict.add_argument("--drugs", type=Path, required=True, help="One supported query drug ID per line")
    predict.add_argument("--anchors", type=Path, help="Optional measured drug-response CSV")
    predict.add_argument("--context", default="target_context", help="Output label for this context")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--csv", action="store_true", help="Also write a full prediction CSV")
    predict.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args(argv)
    try:
        predictor = GRAFTPredictor(args.bundle, device=args.device)
        result = predictor.predict_files(args.genetic, args.drugs, anchors=args.anchors, context=args.context)
        result.save(args.output, csv=args.csv)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"GRAFT: {error}", file=sys.stderr)
        return 2
    print(f"Saved {len(result.drug_ids)} drug profiles × {len(result.genes)} genes "
          f"({result.metadata['mode']}, {len(result.metadata['seeds'])} seeds) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
