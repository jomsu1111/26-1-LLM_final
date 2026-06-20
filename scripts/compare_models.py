#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


BASE_METHODS = ["Always STOP", "Always VERIFY", "Always SC-3", "Oracle Router"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model-level evaluation summaries.")
    parser.add_argument(
        "--model_eval",
        action="append",
        nargs=2,
        metavar=("MODEL_LABEL", "EVAL_DIR"),
        required=True,
        help="Pair of model label and eval directory containing results_summary.csv. Can be repeated.",
    )
    parser.add_argument("--output_csv", default="outputs/model_comparison.csv")
    parser.add_argument("--output_plot", default="outputs/model_comparison_accuracy_cost.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for model_label, eval_dir in args.model_eval:
        summary_path = Path(eval_dir) / "results_summary.csv"
        summary = pd.read_csv(summary_path)
        selected = select_methods(summary)
        selected.insert(0, "model", model_label)
        rows.append(selected)

    comparison = pd.concat(rows, ignore_index=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)

    plot_accuracy_vs_cost(comparison, Path(args.output_plot))
    print(f"Wrote {output_csv}")
    print(f"Wrote {args.output_plot}")


def select_methods(summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for method in BASE_METHODS:
        rows = summary[summary["method"] == method]
        if not rows.empty:
            selected.append(rows.iloc[0])

    controller_rows = summary[summary["method"].astype(str).str.startswith("Controller")]
    if not controller_rows.empty:
        best_idx = controller_rows["utility"].astype(float).idxmax()
        best = controller_rows.loc[best_idx].copy()
        best["method"] = "Best learned controller"
        best["source_method"] = controller_rows.loc[best_idx, "method"]
        selected.append(best)

    result = pd.DataFrame(selected)
    if "source_method" not in result.columns:
        result["source_method"] = result["method"]
    else:
        result["source_method"] = result["source_method"].fillna(result["method"])
    return result


def plot_accuracy_vs_cost(comparison: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, (model, group) in enumerate(comparison.groupby("model")):
        marker = markers[idx % len(markers)]
        ax.scatter(group["avg_total_tokens"], group["accuracy"], label=model, marker=marker, s=70)
        for _, row in group.iterrows():
            label = str(row["method"]).replace("Always ", "")
            ax.annotate(label, (row["avg_total_tokens"], row["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Average total tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison: Accuracy vs Token Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()

