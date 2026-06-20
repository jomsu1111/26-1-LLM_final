#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cost import infer_cost_normalizer
from src.oracle import oracle_per_example, summarize_methods
from src.utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Build simple oracle analysis over rollout JSONL.")
    parser.add_argument("--rollout_path", required=True)
    parser.add_argument("--lambda_values", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--cost_normalizer", type=float, default=None)
    parser.add_argument("--output_dir", default="outputs/oracle")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [row for row in read_jsonl(args.rollout_path) if not row.get("error")]
    if not rows:
        raise ValueError(f"No successful rollout rows found in {args.rollout_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = infer_cost_normalizer(rows, args.cost_normalizer)
    summaries = [
        summarize_methods(rows, lambda_value=lambda_value, normalizer=normalizer)
        for lambda_value in args.lambda_values
    ]
    analysis = pd.concat(summaries, ignore_index=True)
    analysis["cost_normalizer"] = normalizer
    analysis_path = output_dir / "oracle_analysis.csv"
    analysis.to_csv(analysis_path, index=False)

    per_example = oracle_per_example(rows, args.lambda_values, normalizer=normalizer)
    per_example["cost_normalizer"] = normalizer
    per_example_path = output_dir / "oracle_per_example.csv"
    per_example.to_csv(per_example_path, index=False)

    print(f"Wrote {analysis_path}")
    print(f"Wrote {per_example_path}")


if __name__ == "__main__":
    main()

