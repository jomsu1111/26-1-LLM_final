import time
from dataclasses import dataclass
from typing import Optional

from src.utils import whitespace_token_count


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency: float


class MockGenerator:
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        start = time.time()
        guess = _mock_answer_from_prompt(prompt)
        text = f"We solve the problem with a lightweight mock response.\nFinal answer: {guess}"
        input_tokens = whitespace_token_count(prompt)
        output_tokens = whitespace_token_count(text)
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency=time.time() - start,
        )


class HFGenerator:
    def __init__(self, model_name: str, seed: int = 42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.seed = seed

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> GenerationResult:
        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt
        inputs = self.tokenizer(input_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_tokens = int(generated_ids.shape[-1])
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency=time.time() - start,
        )


def load_generator(model_name: str, seed: int = 42, mock_model: bool = False):
    if mock_model:
        return MockGenerator()
    return HFGenerator(model_name=model_name, seed=seed)


def _mock_answer_from_prompt(prompt: str) -> str:
    # Deterministic and intentionally simple. This keeps smoke tests cheap.
    import hashlib

    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return str(int(digest[:4], 16) % 100)

