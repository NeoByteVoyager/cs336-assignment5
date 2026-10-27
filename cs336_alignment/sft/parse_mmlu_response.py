import re
from typing import Any


def parse_mmlu_response(
    mmlu_example: dict[str, Any] | None,
    model_output: str
) -> str | None:

    match = re.search(
        r"\b(?:the correct answer is|the answer is|the ans is)\s+([a-d])\b",
        model_output,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None