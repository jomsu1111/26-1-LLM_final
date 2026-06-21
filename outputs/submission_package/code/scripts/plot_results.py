#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation and oracle analysis results.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/plots")
    parser.add_argument("--oracle_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(eval_dir / "results_summary.csv")
    action_dist = pd.read_csv(eval_dir / "action_distribution.csv")
    transition = pd.read_csv(eval_dir / "transition_table.csv")

    plot_accuracy_vs_cost(summary, output_dir)
    plot_controller_action_distribution(action_dist, output_dir)
    plot_transition_table(transition, output_dir)

    oracle_dir = Path(args.oracle_dir) if args.oracle_dir else None
    if oracle_dir and (oracle_dir / "oracle_analysis.csv").exists():
        oracle = pd.read_csv(oracle_dir / "oracle_analysis.csv")
        plot_utility_vs_lambda(oracle, output_dir)
        plot_oracle_action_distribution(oracle, output_dir)

    print(f"Wrote plots to {output_dir}")


def plot_accuracy_vs_cost(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in summary.iterrows():
        ax.scatter(row["avg_total_tokens"], row["accuracy"], s=70)
        ax.annotate(row["method"], (row["avg_total_tokens"], row["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Average total tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Average Token Cost")
    ax.grid(True, alpha=0.3)
    savefig(fig, output_dir / "accuracy_vs_token_cost")


def plot_utility_vs_lambda(oracle: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in oracle.groupby("method"):
        group = group.sort_values("lambda_value")
        ax.plot(group["lambda_value"], group["utility"], marker="o", label=method)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Utility")
    ax.set_title("Utility vs Lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, output_dir / "utility_vs_lambda")


def plot_oracle_action_distribution(oracle: pd.DataFrame, output_dir: Path) -> None:
    oracle_rows = oracle[oracle["method"] == "Oracle"].sort_values("lambda_value")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(oracle_rows))
    bottom = [0] * len(oracle_rows)
    for action, column in [("STOP", "stop_count"), ("VERIFY", "verify_count"), ("SC-3", "sc3_count")]:
        values = list(oracle_rows[column])
        ax.bar(x, values, bottom=bottom, label=action)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in oracle_rows["lambda_value"]])
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Count")
    ax.set_title("Oracle Action Distribution vs Lambda")
    ax.legend()
    savefig(fig, output_dir / "oracle_action_distribution")


def plot_controller_action_distribution(action_dist: pd.DataFrame, output_dir: Path) -> None:
    pivot = action_dist.pivot(index="method", columns="action", values="fraction").fillna(0.0)
    for action in ["STOP", "VERIFY", "SC-3"]:
        if action not in pivot.columns:
            pivot[action] = 0.0
    pivot = pivot[["STOP", "VERIFY", "SC-3"]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.45)))
    left = [0.0] * len(pivot)
    y = range(len(pivot))
    for action in ["STOP", "VERIFY", "SC-3"]:
        values = list(pivot[action])
        ax.barh(y, values, left=left, label=action)
        left = [l + v for l, v in zip(left, values)]
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_xlabel("Fraction")
    ax.set_title("Action Distribution")
    ax.legend(loc="lower right")
    savefig(fig, output_dir / "controller_action_distribution")


def plot_transition_table(transition: pd.DataFrame, output_dir: Path) -> None:
    labels = list(transition["transition"])
    values = list(transition["rate"])
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Transition Rates")
    ax.tick_params(axis="x", labelrotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    savefig(fig, output_dir / "transition_table")


def savefig(fig, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=200)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()

