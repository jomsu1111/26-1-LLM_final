#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import load_reasoning_dataset
from src.generation import error_row, run_example_rollout
from src.model_utils import load_generator
from src.utils import append_jsonl, load_completed_ids, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run initial, STOP, VERIFY, and SC-3 rollouts.")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before applying max_examples.")
    parser.add_argument("--output_path", default="outputs/gsm8k_rollouts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens_initial", type=int, default=256)
    parser.add_argument("--max_new_tokens_verify", type=int, default=256)
    parser.add_argument("--max_new_tokens_sc3", type=int, default=256)
    parser.add_argument("--mock_model", action="store_true", help="Use a deterministic cheap mock generator.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cache and rewrite output file.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.overwrite and Path(args.output_path).exists():
        Path(args.output_path).unlink()

    completed_ids = load_completed_ids(args.output_path)
    examples = load_reasoning_dataset(
        args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    generator = load_generator(args.model_name, seed=args.seed, mock_model=args.mock_model)

    for example in tqdm(examples, desc="rollouts"):
        if example.example_id in completed_ids:
            continue
        try:
            row = run_example_rollout(
                example=example,
                generator=generator,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens_initial=args.max_new_tokens_initial,
                max_new_tokens_verify=args.max_new_tokens_verify,
                max_new_tokens_sc3=args.max_new_tokens_sc3,
            )
        except Exception as exc:
            row = error_row(example, exc)
        append_jsonl(args.output_path, row)

    print(f"Wrote rollouts to {args.output_path}")


if __name__ == "__main__":
    main()
