#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers import load_controller
from src.cost import infer_cost_normalizer
from src.evaluation import (
    action_distribution,
    build_method_actions,
    evaluate_action_policy,
    per_example_predictions,
    transition_table,
)
from src.splits import add_split
from src.utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fixed baselines, controller, and oracle router.")
    parser.add_argument("--rollout_path", required=True)
    parser.add_argument("--controller_path", action="append", default=None, help="Path to a saved controller. Can be repeated.")
    parser.add_argument("--lambda_value", type=float, required=True)
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--eval_split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--cost_normalizer", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = [
        add_split(row, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
        for row in read_jsonl(args.rollout_path)
        if not row.get("error")
    ]
    if not all_rows:
        raise ValueError(f"No successful rollout rows found in {args.rollout_path}")
    rows = all_rows if args.eval_split == "all" else [row for row in all_rows if row["split"] == args.eval_split]
    if not rows:
        raise ValueError(f"No rows found for eval_split={args.eval_split}. Increase rollout size or use --eval_split all.")

    normalizer = infer_cost_normalizer(all_rows, args.cost_normalizer)
    method_actions = build_method_actions(rows, args.lambda_value, normalizer, seed=args.seed)

    if args.controller_path:
        for controller_path in args.controller_path:
            controller = load_controller(controller_path)
            method_name = _controller_method_name(controller)
            method_actions[method_name] = controller.predict_actions(rows)

    summary = pd.DataFrame.from_records(
        [
            evaluate_action_policy(rows, method, actions, args.lambda_value, normalizer)
            for method, actions in method_actions.items()
        ]
    )
    summary["eval_split"] = args.eval_split
    summary["cost_normalizer"] = normalizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "results_summary.csv"
    per_example_path = output_dir / "per_example_predictions.csv"
    action_dist_path = output_dir / "action_distribution.csv"
    transition_path = output_dir / "transition_table.csv"

    summary.to_csv(summary_path, index=False)
    per_example_predictions(rows, method_actions, args.lambda_value, normalizer).to_csv(per_example_path, index=False)
    action_distribution(summary).to_csv(action_dist_path, index=False)
    transition_table(rows).to_csv(transition_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_example_path}")
    print(f"Wrote {action_dist_path}")
    print(f"Wrote {transition_path}")


def _controller_method_name(controller) -> str:
    name = f"Controller ({controller.controller_type}, {controller.classifier_type})"
    if getattr(controller, "calibration_thresholds", None):
        name += " calibrated"
    return name


if __name__ == "__main__":
    main()
