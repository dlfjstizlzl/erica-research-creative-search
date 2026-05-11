"""Problem reframing for clearer search inputs."""

from __future__ import annotations
import random

from config import (
    DEFAULT_OUTPUT_LANGUAGE,
    PROBLEM_REFRAMING_PROMPT,
    REFRAMER_MODELS,
    REFRAMER_NUM_PREDICT,
)
from core.utils import load_text
from llm.ollama_client import AsyncOllamaClient


async def reframe_problem(
    problem: str,
    *,
    model: str | None = None,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """Rewrite a raw problem into a clearer search input using async LLM call."""
    prompt_template = load_text(PROBLEM_REFRAMING_PROMPT)
    prompt = prompt_template.format(problem=problem, language=language)
    selected_model = model or random.choice(REFRAMER_MODELS)
    client = AsyncOllamaClient(model=selected_model)

    print(f"[reframer] Reframing problem with model={selected_model}")
    payload = await client.chat_json(
        user_prompt=prompt,
        system_prompt=(
            "You rewrite user problems into clearer search inputs. "
            "Preserve intent. "
            "Do not solve the problem. "
            "Do not introduce constraints or extra sections. "
            "Return exactly one JSON object with reframed_problem. "
            "Return only valid JSON."
        ),
        debug_label=f"reframer model={selected_model}",
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
