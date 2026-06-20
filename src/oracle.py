from collections import Counter
from typing import Dict, Iterable, List

import pandas as pd

from src.cost import ACTIONS, action_additional_tokens, action_correct, action_latency, action_total_tokens, utility


def oracle_action(row: Dict, lambda_value: float, normalizer: float) -> str:
    scores = {action: utility(row, action, lambda_value, normalizer) for action in ACTIONS}
    # Stable tie-break: prefer cheaper actions in the natural order.
    return max(ACTIONS, key=lambda action: (scores[action], -action_total_tokens(row, action)))


def summarize_methods(rows: List[Dict], lambda_value: float, normalizer: float) -> pd.DataFrame:
    records = []
    methods = list(ACTIONS) + ["Oracle"]
    for method in methods:
        chosen = [oracle_action(row, lambda_value, normalizer) if method == "Oracle" else method for row in rows]
        correct = [action_correct(row, action) for row, action in zip(rows, chosen)]
        total_tokens = [action_total_tokens(row, action) for row, action in zip(rows, chosen)]
        additional_tokens = [action_additional_tokens(row, action) for row, action in zip(rows, chosen)]
        latency = [action_latency(row, action) for row, action in zip(rows, chosen)]
        utilities = [utility(row, action, lambda_value, normalizer) for row, action in zip(rows, chosen)]
        distribution = Counter(chosen)
        records.append(
            {
                "lambda_value": lambda_value,
                "method": method,
                "accuracy": sum(correct) / max(len(correct), 1),
                "avg_total_tokens": sum(total_tokens) / max(len(total_tokens), 1),
                "avg_additional_tokens": sum(additional_tokens) / max(len(additional_tokens), 1),
                "avg_latency": sum(latency) / max(len(latency), 1),
                "utility": sum(utilities) / max(len(utilities), 1),
                "num_examples": len(rows),
                "stop_count": distribution.get("STOP", 0),
                "verify_count": distribution.get("VERIFY", 0),
                "sc3_count": distribution.get("SC-3", 0),
            }
        )
    return pd.DataFrame.from_records(records)


def oracle_per_example(rows: Iterable[Dict], lambda_values: List[float], normalizer: float) -> pd.DataFrame:
    records = []
    for row in rows:
        for lambda_value in lambda_values:
            action = oracle_action(row, lambda_value, normalizer)
            records.append(
                {
                    "example_id": row.get("example_id"),
                    "lambda_value": lambda_value,
                    "oracle_action": action,
                    "oracle_correct": action_correct(row, action),
                    "oracle_total_tokens": action_total_tokens(row, action),
                    "oracle_utility": utility(row, action, lambda_value, normalizer),
                }
            )
    return pd.DataFrame.from_records(records)

