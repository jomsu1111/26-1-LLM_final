import math
import re
from decimal import Decimal, InvalidOperation
from typing import Optional


FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    numeric = extract_numeric_answer(answer_text)
    return numeric or answer_text.strip()


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = FINAL_ANSWER_RE.search(text)
    if match:
        answer = match.group(1).strip()
        return answer.splitlines()[0].strip()
    numeric = extract_numeric_answer(text)
    return numeric or ""


def extract_numeric_answer(text: str) -> Optional[str]:
    if not text:
        return None
    matches = NUMBER_RE.findall(text.replace("$", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = answer.replace(",", "").replace("$", "")
    answer = re.sub(r"\s+", " ", answer)
    numeric = extract_numeric_answer(answer)
    if numeric is not None:
        return normalize_number(numeric)
    return answer.lower().strip(" .")


def normalize_number(value: str) -> str:
    try:
        dec = Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return value.strip()
    if dec == dec.to_integral_value():
        return str(int(dec))
    return format(dec.normalize(), "f").rstrip("0").rstrip(".")


def is_correct(predicted: str, gold: str) -> bool:
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        return math.isclose(float(pred_norm), float(gold_norm), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        return False


def has_valid_final_answer(text: str) -> bool:
    return bool(FINAL_ANSWER_RE.search(text or ""))

