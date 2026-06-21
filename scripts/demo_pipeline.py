#!/usr/bin/env python3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    print("Stop, Verify, or Explore? - Demo Pipeline", flush=True)
    print("This demo reuses a cached mock rollout and regenerates oracle/evaluation CSV files.", flush=True)

    run(
        [
            "python3",
            "scripts/build_oracle.py",
            "--rollout_path",
            "outputs/gsm8k_rollouts_mock.jsonl",
            "--lambda_values",
            "0.1",
            "--output_dir",
            "outputs/demo_oracle_mock",
        ]
    )
    run(
        [
            "python3",
            "scripts/evaluate_methods.py",
            "--rollout_path",
            "outputs/gsm8k_rollouts_mock.jsonl",
            "--lambda_value",
            "0.1",
            "--output_dir",
            "outputs/demo_eval_mock",
            "--eval_split",
            "all",
        ]
    )

    summary = ROOT / "outputs/demo_eval_mock/results_summary.csv"
    print("\nGenerated result summary:", flush=True)
    print(summary, flush=True)
    print(summary.read_text(encoding="utf-8"), flush=True)

    print("Demo complete.", flush=True)


if __name__ == "__main__":
    main()
