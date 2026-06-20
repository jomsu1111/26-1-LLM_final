import re
from typing import Dict, List

from src.answer_extraction import extract_numeric_answer, has_valid_final_answer
from src.utils import whitespace_token_count


INPUT_ONLY_FEATURES = [
    "question_token_length",
    "question_word_count",
    "question_numeric_token_count",
    "question_arithmetic_symbol_count",
]

STATE_AWARE_FEATURES = INPUT_ONLY_FEATURES + [
    "initial_reasoning_token_length",
    "answer_format_validity",
    "numeric_answer_extracted",
    "initial_input_tokens",
    "initial_output_tokens",
    "initial_total_tokens",
]


def build_basic_features(question: str, initial_trace: str, initial_answer: str) -> Dict[str, object]:
    return {
        "question_token_length": whitespace_token_count(question),
        "question_word_count": len(re.findall(r"\w+", question or "")),
        "question_numeric_token_count": len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", question or "")),
        "question_arithmetic_symbol_count": len(re.findall(r"[+\-*/=]", question or "")),
        "initial_reasoning_token_length": whitespace_token_count(initial_trace),
        "answer_format_validity": has_valid_final_answer(initial_trace),
        "numeric_answer_extracted": extract_numeric_answer(initial_answer) is not None,
        "avg_token_logprob": None,
        "avg_token_entropy": None,
        "max_token_entropy": None,
        "self_reported_confidence": None,
    }


def feature_names(controller_type: str) -> List[str]:
    if controller_type == "input_only":
        return INPUT_ONLY_FEATURES
    if controller_type == "state_aware":
        return STATE_AWARE_FEATURES
    raise ValueError(f"Unknown controller_type: {controller_type}")


def row_to_features(row: Dict, controller_type: str) -> List[float]:
    names = feature_names(controller_type)
    values = []
    for name in names:
        value = row.get(name)
        if value is None:
            value = _fallback_feature(row, name)
        values.append(_to_float(value))
    return values


def _fallback_feature(row: Dict, name: str):
    if name in INPUT_ONLY_FEATURES:
        question = row.get("question", "")
        return build_basic_features(question, "", "").get(name)
    if name == "initial_reasoning_token_length":
        return whitespace_token_count(row.get("initial_trace", ""))
    if name == "answer_format_validity":
        return has_valid_final_answer(row.get("initial_trace", ""))
    if name == "numeric_answer_extracted":
        return extract_numeric_answer(row.get("initial_answer", "")) is not None
    if name in {"initial_input_tokens", "initial_output_tokens", "initial_total_tokens"}:
        return row.get(name, 0)
    return 0


def _to_float(value) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
