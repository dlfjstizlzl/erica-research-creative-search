"""Problem reframing for clearer search inputs."""

from __future__ import annotations

from config import (
    DEFAULT_OUTPUT_LANGUAGE,
    PROBLEM_REFRAMING_PROMPT,
    REFRAMER_MODEL,
    REFRAMER_NUM_PREDICT,
)
from core.utils import load_text
from llm import OllamaClient


def reframe_problem(
    problem: str,
    *,
    model: str = REFRAMER_MODEL,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """Rewrite a raw problem into a clearer search input."""
    prompt_template = load_text(PROBLEM_REFRAMING_PROMPT)
    prompt = prompt_template.format(problem=problem, language=language)
    client = OllamaClient(model=model)

    print(f"[reframer] Reframing problem with model={model}")
    payload = client.chat_json(
        prompt,
        system_prompt=(
            "You rewrite user problems into clearer search inputs. "
            "Preserve intent. "
            "Do not solve the problem. "
            "Do not introduce constraints or extra sections. "
            "Return exactly one JSON object with reframed_problem. "
            "Return only valid JSON."
        ),
        debug_label=f"reframer model={model}",
        num_predict=REFRAMER_NUM_PREDICT,
    )

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        print("[reframer] Unexpected payload. Falling back to raw problem.")
        return problem

    reframed_problem = " ".join(
        str(payload.get("reframed_problem") or "").split()
    ).strip()
    if not reframed_problem:
        print("[reframer] Empty reframed problem. Falling back to raw problem.")
        return problem

    print(f"[reframer] Reframed problem: {reframed_problem}")
    return reframed_problem
