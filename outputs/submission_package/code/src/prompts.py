INITIAL_REASONING_PROMPT = """Solve the grade-school math problem. Show concise reasoning, then end with exactly:
Final answer: <answer>

Problem:
{question}
"""

VERIFY_PROMPT = """You are a careful verifier for a grade-school math problem.

First solve the problem again from scratch without trusting the previous solution.
Then compare your new solution with the previous reasoning and answer.
If the previous answer is correct, keep it. If it is wrong, replace it.
Do not simply repeat the previous reasoning; actively check each arithmetic step.

Problem:
{question}

Previous reasoning and answer:
{initial_trace}

Previous extracted answer: {initial_answer}

Show concise verification reasoning, then end with exactly:
Final answer: <answer>
"""

SC3_PROMPT = """Solve the grade-school math problem independently using a fresh approach.
Try to use a different decomposition, equation, or arithmetic check than a first attempt might use.
Be concise but explicit enough that the final number is justified.
End with exactly:
Final answer: <answer>

Problem:
{question}
"""


def build_initial_prompt(question: str) -> str:
    return INITIAL_REASONING_PROMPT.format(question=question)


def build_verify_prompt(question: str, initial_trace: str, initial_answer: str) -> str:
    return VERIFY_PROMPT.format(
        question=question,
        initial_trace=initial_trace,
        initial_answer=initial_answer,
    )


def build_sc3_prompt(question: str) -> str:
    return SC3_PROMPT.format(question=question)
