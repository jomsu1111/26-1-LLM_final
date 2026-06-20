from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

from src.answer_extraction import extract_gsm8k_gold_answer


@dataclass
class ReasoningExample:
    example_id: str
    dataset: str
    question: str
    gold_answer: str


def load_gsm8k(
    split: str = "test",
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[ReasoningExample]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    examples: List[ReasoningExample] = []
    for idx, row in enumerate(dataset):
        source_idx = row.get("idx", idx)
        examples.append(
            ReasoningExample(
                example_id=f"gsm8k-{split}-{source_idx}",
                dataset="gsm8k",
                question=row["question"],
                gold_answer=extract_gsm8k_gold_answer(row["answer"]),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def load_reasoning_dataset(
    dataset_name: str,
    split: str = "test",
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[ReasoningExample]:
    if dataset_name.lower() == "gsm8k":
        return load_gsm8k(split=split, max_examples=max_examples, shuffle=shuffle, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
