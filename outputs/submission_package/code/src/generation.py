from collections import Counter
from typing import Any, Dict, List

from src.answer_extraction import extract_final_answer, is_correct, normalize_answer
from src.features import build_basic_features
from src.model_utils import GenerationResult
from src.prompts import build_initial_prompt, build_sc3_prompt, build_verify_prompt


def run_example_rollout(
    example,
    generator,
    temperature: float,
    top_p: float,
    max_new_tokens_initial: int,
    max_new_tokens_verify: int,
    max_new_tokens_sc3: int,
) -> Dict[str, Any]:
    initial_prompt = build_initial_prompt(example.question)
    initial = generator.generate(
        initial_prompt,
        max_new_tokens=max_new_tokens_initial,
        temperature=temperature,
        top_p=top_p,
    )
    initial_answer = extract_final_answer(initial.text)
    initial_correct = is_correct(initial_answer, example.gold_answer)

    row: Dict[str, Any] = {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "initial_trace": initial.text,
        "initial_answer": initial_answer,
        "initial_correct": initial_correct,
        "initial_input_tokens": initial.input_tokens,
        "initial_output_tokens": initial.output_tokens,
        "initial_total_tokens": initial.total_tokens,
        "initial_latency": initial.latency,
    }

    row.update(_stop_fields(initial_answer, initial_correct))
    row.update(_verify_fields(example, generator, initial.text, initial_answer, temperature, top_p, max_new_tokens_verify))
    row.update(_sc3_fields(example, generator, initial_answer, temperature, top_p, max_new_tokens_sc3))
    row.update(build_basic_features(example.question, initial.text, initial_answer))
    row["error"] = None
    return row


def _stop_fields(initial_answer: str, initial_correct: bool) -> Dict[str, Any]:
    return {
        "stop_answer": initial_answer,
        "stop_correct": initial_correct,
        "stop_additional_tokens": 0,
        "stop_latency": 0.0,
    }


def _verify_fields(
    example,
    generator,
    initial_trace: str,
    initial_answer: str,
    temperature: float,
    top_p: float,
    max_new_tokens_verify: int,
) -> Dict[str, Any]:
    prompt = build_verify_prompt(example.question, initial_trace, initial_answer)
    result: GenerationResult = generator.generate(
        prompt,
        max_new_tokens=max_new_tokens_verify,
        temperature=temperature,
        top_p=top_p,
    )
    answer = extract_final_answer(result.text)
    return {
        "verify_trace": result.text,
        "verify_answer": answer,
        "verify_correct": is_correct(answer, example.gold_answer),
        "verify_input_tokens": result.input_tokens,
        "verify_output_tokens": result.output_tokens,
        "verify_total_tokens": result.total_tokens,
        "verify_latency": result.latency,
    }


def _sc3_fields(
    example,
    generator,
    initial_answer: str,
    temperature: float,
    top_p: float,
    max_new_tokens_sc3: int,
) -> Dict[str, Any]:
    traces: List[str] = []
    answers: List[str] = []
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    for _ in range(2):
        result: GenerationResult = generator.generate(
            build_sc3_prompt(example.question),
            max_new_tokens=max_new_tokens_sc3,
            temperature=temperature,
            top_p=top_p,
        )
        traces.append(result.text)
        answer = extract_final_answer(result.text)
        answers.append(answer)
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        latency += result.latency

    all_answers = [initial_answer] + answers
    final_answer = plurality_vote(all_answers, tie_break_answer=initial_answer)
    return {
        "sc3_traces": traces,
        "sc3_answers": all_answers,
        "sc3_final_answer": final_answer,
        "sc3_correct": is_correct(final_answer, example.gold_answer),
        "sc3_input_tokens": input_tokens,
        "sc3_output_tokens": output_tokens,
        "sc3_total_tokens": input_tokens + output_tokens,
        "sc3_latency": latency,
    }


def plurality_vote(answers: List[str], tie_break_answer: str) -> str:
    if not answers:
        return ""
    counts = Counter(normalize_answer(answer) for answer in answers)
    if not counts:
        return tie_break_answer
    max_count = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == max_count}
    if normalize_answer(tie_break_answer) in winners:
        return tie_break_answer
    for answer in answers:
        if normalize_answer(answer) in winners:
            return answer
    return tie_break_answer


def error_row(example, error: Exception) -> Dict[str, Any]:
    return {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "error": repr(error),
    }

