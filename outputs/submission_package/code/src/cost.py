from typing import Dict, Iterable, Optional


ACTIONS = ("STOP", "VERIFY", "SC-3")


def action_correct(row: Dict, action: str) -> int:
    if action == "STOP":
        return int(bool(row.get("stop_correct")))
    if action == "VERIFY":
        return int(bool(row.get("verify_correct")))
    if action == "SC-3":
        return int(bool(row.get("sc3_correct")))
    raise ValueError(f"Unknown action: {action}")


def action_total_tokens(row: Dict, action: str) -> int:
    if action == "STOP":
        return int(row.get("initial_total_tokens") or 0)
    if action == "VERIFY":
        return int(row.get("initial_total_tokens") or 0) + int(row.get("verify_total_tokens") or 0)
    if action == "SC-3":
        return int(row.get("initial_total_tokens") or 0) + int(row.get("sc3_total_tokens") or 0)
    raise ValueError(f"Unknown action: {action}")


def action_additional_tokens(row: Dict, action: str) -> int:
    if action == "STOP":
        return 0
    if action == "VERIFY":
        return int(row.get("verify_total_tokens") or 0)
    if action == "SC-3":
        return int(row.get("sc3_total_tokens") or 0)
    raise ValueError(f"Unknown action: {action}")


def action_latency(row: Dict, action: str) -> float:
    initial = float(row.get("initial_latency") or 0.0)
    if action == "STOP":
        return initial
    if action == "VERIFY":
        return initial + float(row.get("verify_latency") or 0.0)
    if action == "SC-3":
        return initial + float(row.get("sc3_latency") or 0.0)
    raise ValueError(f"Unknown action: {action}")


def infer_cost_normalizer(rows: Iterable[Dict], explicit: Optional[float] = None) -> float:
    if explicit is not None and explicit > 0:
        return explicit
    costs = [action_total_tokens(row, "SC-3") for row in rows if not row.get("error")]
    avg = sum(costs) / max(len(costs), 1)
    return avg if avg > 0 else 1.0


def utility(row: Dict, action: str, lambda_value: float, normalizer: float) -> float:
    return action_correct(row, action) - lambda_value * (action_total_tokens(row, action) / normalizer)

