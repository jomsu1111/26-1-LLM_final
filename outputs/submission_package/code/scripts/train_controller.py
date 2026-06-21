#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers import save_controller, train_oracle_action_classifier, train_value_predictor
from src.cost import infer_cost_normalizer
from src.splits import add_split
from src.utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight oracle-action controller.")
    parser.add_argument("--rollout_path", required=True)
    parser.add_argument("--lambda_value", type=float, required=True)
    parser.add_argument("--controller_type", choices=["input_only", "state_aware"], default="state_aware")
    parser.add_argument("--classifier_type", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--controller_variant", choices=["oracle_classifier", "value_predictor"], default="oracle_classifier")
    parser.add_argument("--calibrate", action="store_true", help="Tune extra-compute confidence thresholds on validation split.")
    parser.add_argument("--output_dir", default="outputs/controllers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--cost_normalizer", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [
        add_split(row, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
        for row in read_jsonl(args.rollout_path)
        if not row.get("error")
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows:
        raise ValueError("No train rows available. Increase rollout size or adjust split fractions.")

    normalizer = infer_cost_normalizer(rows, args.cost_normalizer)
    if args.controller_variant == "value_predictor":
        bundle = train_value_predictor(
            rows=train_rows,
            controller_type=args.controller_type,
            lambda_value=args.lambda_value,
            cost_normalizer=normalizer,
            seed=args.seed,
        )
    else:
        bundle = train_oracle_action_classifier(
            rows=train_rows,
            controller_type=args.controller_type,
            classifier_type=args.classifier_type,
            lambda_value=args.lambda_value,
            cost_normalizer=normalizer,
            seed=args.seed,
            validation_rows=validation_rows,
            calibrate=args.calibrate,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_tag = str(args.lambda_value)
    calibrate_tag = "_calibrated" if args.calibrate and args.controller_variant == "oracle_classifier" else ""
    model_path = output_dir / f"{args.controller_type}_{bundle.classifier_type}{calibrate_tag}_lambda_{lambda_tag}.pkl"
    save_controller(bundle, str(model_path))

    metadata = {
        "controller_path": str(model_path),
        "rollout_path": args.rollout_path,
        "lambda_value": args.lambda_value,
        "cost_normalizer": normalizer,
        "controller_type": args.controller_type,
        "controller_variant": args.controller_variant,
        "classifier_type": bundle.classifier_type,
        "calibrated": bool(bundle.calibration_thresholds),
        "calibration_thresholds": bundle.calibration_thresholds,
        "validation_utility": bundle.validation_utility,
        "validation_action_counts": bundle.validation_action_counts,
        "seed": args.seed,
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_validation_rows": len(validation_rows),
        "label_counts": bundle.label_counts,
        "feature_names": bundle.feature_names,
    }
    metadata_path = output_dir / f"{args.controller_type}_{bundle.classifier_type}{calibrate_tag}_lambda_{lambda_tag}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {model_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
