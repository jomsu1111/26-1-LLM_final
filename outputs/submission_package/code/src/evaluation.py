from collections import Counter
from typing import Dict, List

import pandas as pd

from src.controllers import length_threshold_actions, random_actions
from src.cost import (
    ACTIONS,
    action_additional_tokens,
    action_correct,
    action_latency,
    action_total_tokens,
    utility,
)
from src.oracle import oracle_action


def evaluate_action_policy(
    rows: List[Dict],
    method: str,
    actions: List[str],
    lambda_value: float,
    cost_normalizer: float,
) -> Dict:
    correct = [action_correct(row, action) for row, action in zip(rows, actions)]
    total_tokens = [action_total_tokens(row, action) for row, action in zip(rows, actions)]
    additional_tokens = [action_additional_tokens(row, action) for row, action in zip(rows, actions)]
    latencies = [action_latency(row, action) for row, action in zip(rows, actions)]
    utilities = [utility(row, action, lambda_value, cost_normalizer) for row, action in zip(rows, actions)]
    counts = Counter(actions)
    return {
        "method": method,
        "lambda_value": lambda_value,
        "accuracy": sum(correct) / max(len(correct), 1),
        "avg_total_tokens": sum(total_tokens) / max(len(total_tokens), 1),
        "avg_additional_tokens": sum(additional_tokens) / max(len(additional_tokens), 1),
        "avg_latency": sum(latencies) / max(len(latencies), 1),
        "utility": sum(utilities) / max(len(utilities), 1),
        "num_examples": len(rows),
        "stop_count": counts.get("STOP", 0),
        "verify_count": counts.get("VERIFY", 0),
        "sc3_count": counts.get("SC-3", 0),
    }


def per_example_predictions(
    rows: List[Dict],
    method_actions: Dict[str, List[str]],
    lambda_value: float,
    cost_normalizer: float,
) -> pd.DataFrame:
    records = []
    for idx, row in enumerate(rows):
        for method, actions in method_actions.items():
            action = actions[idx]
            records.append(
                {
                    "example_id": row.get("example_id"),
                    "split": row.get("split"),
                    "method": method,
                    "lambda_value": lambda_value,
                    "action": action,
                    "correct": action_correct(row, action),
                    "total_tokens": action_total_tokens(row, action),
                    "additional_tokens": action_additional_tokens(row, action),
                    "latency": action_latency(row, action),
                    "utility": utility(row, action, lambda_value, cost_normalizer),
                    "gold_answer": row.get("gold_answer"),
                    "initial_answer": row.get("initial_answer"),
                }
            )
    return pd.DataFrame.from_records(records)


def action_distribution(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in summary.iterrows():
        denom = max(int(row["num_examples"]), 1)
        records.extend(
            [
                {"method": row["method"], "action": "STOP", "count": row["stop_count"], "fraction": row["stop_count"] / denom},
                {
                    "method": row["method"],
                    "action": "VERIFY",
                    "count": row["verify_count"],
                    "fraction": row["verify_count"] / denom,
                },
                {"method": row["method"], "action": "SC-3", "count": row["sc3_count"], "fraction": row["sc3_count"] / denom},
            ]
        )
    return pd.DataFrame.from_records(records)


def build_method_actions(rows: List[Dict], lambda_value: float, cost_normalizer: float, seed: int = 42) -> Dict[str, List[str]]:
    return {
        "Always STOP": ["STOP"] * len(rows),
        "Always VERIFY": ["VERIFY"] * len(rows),
        "Always SC-3": ["SC-3"] * len(rows),
        "Random Router": random_actions(rows, seed=seed),
        "Length Threshold Router": length_threshold_actions(rows),
        "Oracle Router": [oracle_action(row, lambda_value, cost_normalizer) for row in rows],
    }


def transition_table(rows: List[Dict]) -> pd.DataFrame:
    initial_wrong = [row for row in rows if not row.get("initial_correct")]
    initial_right = [row for row in rows if row.get("initial_correct")]
    records = [
        {
            "transition": "initial_wrong_verify_correct",
            "rate": _rate(initial_wrong, "verify_correct", True),
            "denominator": len(initial_wrong),
        },
        {
            "transition": "initial_right_verify_wrong",
            "rate": _rate(initial_right, "verify_correct", False),
            "denominator": len(initial_right),
        },
        {
            "transition": "initial_wrong_sc3_correct",
            "rate": _rate(initial_wrong, "sc3_correct", True),
            "denominator": len(initial_wrong),
        },
        {
            "transition": "initial_right_sc3_wrong",
            "rate": _rate(initial_right, "sc3_correct", False),
            "denominator": len(initial_right),
        },
    ]
    return pd.DataFrame.from_records(records)


def _rate(rows: List[Dict], key: str, target: bool) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get(key)) is target for row in rows) / len(rows)

