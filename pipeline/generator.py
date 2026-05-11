"""Base idea generation."""

from __future__ import annotations

import asyncio
import random
import time
import uuid

from config import (
    BASE_GENERATION_PROMPT,
    DEFAULT_OUTPUT_LANGUAGE,
    GENERATOR_PERSONAS,
    GENERATOR_NUM_PREDICT,
    GENERATOR_SAMPLE_COUNT,
    OLLAMA_GENERATOR_MODELS,
)
from core.utils import load_text
from llm.ollama_client import AsyncOllamaClient


async def generate_base_ideas(
    problem: str,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict[str, str]]:
    """Generate raw candidate ideas from random persona-model assignments using async I/O."""
    prompt_template = load_text(BASE_GENERATION_PROMPT)
    pairings = _select_generation_requests()
    ideas_with_index: list[tuple[int, dict[str, str]]] = []

    print(f"[generator] Selected {len(pairings)} generation requests.")

    tasks = [
        _generate_single_idea(
            problem=problem,
            language=language,
            prompt_template=prompt_template,
            model=model,
            persona=persona,
            index=index,
            total_requests=len(pairings),
        )
        for index, (model, persona) in enumerate(pairings, start=1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            print(f"[generator] Request failed: {res}")
        elif res is not None:
            ideas_with_index.append(res)

    ideas_with_index.sort(key=lambda item: item[0])
    return [idea for _, idea in ideas_with_index]


async def _generate_single_idea(
    *,
    problem: str,
    language: str,
    prompt_template: str,
    model: str,
    persona: str,
    index: int,
    total_requests: int,
) -> tuple[int, dict[str, str]] | None:
    persona_name = persona.split(":", 1)[0].strip()
    print(
        f"[generator] Request {index}/{total_requests} "
        f"using model={model} persona={persona_name}"
    )
    started_at = time.perf_counter()
    prompt = prompt_template.format(
        problem=problem,
        persona=persona,
        language=language,
    )
    client = AsyncOllamaClient(model=model)
    payload = await client.chat_json(
        user_prompt=prompt,
        system_prompt=(
            "You generate one high-quality strategy idea. "
            "Stay faithful to the assigned persona. "
            "Return exactly one JSON object. "
            "Novelty matters, but avoid gratuitous speculative technology and science-fiction framing. "
            "It is acceptable if the idea is unconventional or partially unrealistic, but it should still read like a legible operating model. "
            "Avoid filtering toward safe, ordinary, or generic answers. "
            "Do not collapse back into generic recommendation, assistant, dashboard, planner, or companion patterns "
            "unless that direction is clearly the most original and concrete option. "
            "Prefer present-day actors, incentives, workflows, institutions, and infrastructure unless frontier technology is clearly justified by the problem. "
            "If you mention advanced technology, explain it in a plausible near-term way instead of relying on quantum, holographic, magical, or vaguely futuristic language. "
            "The description should be concrete and moderately detailed. "
            "Explain mechanism, practical operation, and strategic difference. "
            f"Write all natural-language fields in {language}. "
            "Populate all requested fields with specific, usable content. "
            "Return only valid JSON."
        ),
        debug_label=f"generator#{index} model={model} persona={persona_name}",
        num_predict=GENERATOR_NUM_PREDICT,
    )
    if not payload:
        return None
        
    idea = _normalize_idea(
        payload,
        index=index,
        model=model,
        persona=persona,
    )
    elapsed = time.perf_counter() - started_at
    if idea["description"]:
        print(
            f"[generator] Completed request {index} in {elapsed:.1f}s "
            f"-> {idea['title']}"
        )
        return index, idea

    print(f"[generator] Request {index} returned an empty idea in {elapsed:.1f}s.")
    return None


def _select_generation_requests() -> list[tuple[str, str]]:
    if not OLLAMA_GENERATOR_MODELS:
        raise RuntimeError("No generator models configured.")
    if not GENERATOR_PERSONAS:
        raise RuntimeError("No generator personas configured.")

    if GENERATOR_SAMPLE_COUNT <= len(GENERATOR_PERSONAS):
        personas = random.sample(GENERATOR_PERSONAS, GENERATOR_SAMPLE_COUNT)
    else:
        personas = list(GENERATOR_PERSONAS)
        while len(personas) < GENERATOR_SAMPLE_COUNT:
            personas.append(random.choice(GENERATOR_PERSONAS))

    requests: list[tuple[str, str]] = []
    model_pool = list(OLLAMA_GENERATOR_MODELS)
    random.shuffle(model_pool)

    guaranteed_count = min(len(personas), len(model_pool))
    for index in range(guaranteed_count):
        requests.append((model_pool[index], personas[index]))

    for persona in personas[guaranteed_count:]:
        requests.append((random.choice(OLLAMA_GENERATOR_MODELS), persona))

    random.shuffle(requests)
    return requests


def _normalize_idea(
    payload: object,
    *,
    index: int,
    model: str,
    persona: str,
) -> dict[str, str]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object for a base idea.")

    persona_name = persona.split(":", 1)[0].strip()
    return {
        "id": str(payload.get("id") or uuid.uuid4().hex[:8]),
        "title": str(payload.get("title") or f"Idea {index}").strip(),
        "persona": str(payload.get("persona") or persona_name).strip(),
        "strategy_type": str(payload.get("strategy_type") or "general").strip(),
        "description": " ".join(str(payload.get("description") or "").split()),
        "mechanism": " ".join(str(payload.get("mechanism") or "").split()),
        "target_user": " ".join(str(payload.get("target_user") or "").split()),
        "execution_context": " ".join(
            str(payload.get("execution_context") or "").split()
        ),
        "expected_advantage": " ".join(
            str(payload.get("expected_advantage") or "").split()
        ),
        "source_model": model.strip(),
        "source_persona": persona_name,
    }
