"""Base idea generation."""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BASE_GENERATION_PROMPT,
    DEFAULT_OUTPUT_LANGUAGE,
    GENERATOR_MAX_WORKERS,
    GENERATOR_PERSONAS,
    GENERATOR_NUM_PREDICT,
    GENERATOR_SAMPLE_COUNT,
    OLLAMA_GENERATOR_MODELS,
)
from core.utils import load_text
from llm import OllamaClient


def generate_base_ideas(
    problem: str,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict[str, str]]:
    """Generate five raw candidate ideas from random persona-model assignments."""
    prompt_template = load_text(BASE_GENERATION_PROMPT)
    pairings = _select_generation_requests()
    ideas_with_index: list[tuple[int, dict[str, str]]] = []

    print(f"[generator] Selected {len(pairings)} generation requests.")
    print(
        f"[generator] Running requests in parallel with "
        f"max_workers={min(GENERATOR_MAX_WORKERS, len(pairings))}"
    )

    with ThreadPoolExecutor(
        max_workers=min(GENERATOR_MAX_WORKERS, len(pairings))
    ) as executor:
        futures = [
            executor.submit(
                _generate_single_idea,
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

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f"[generator] Request failed: {exc}")
                continue
            if result is None:
                continue
            ideas_with_index.append(result)

    ideas_with_index.sort(key=lambda item: item[0])
    return [idea for _, idea in ideas_with_index]


def _generate_single_idea(
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
    client = OllamaClient(model=model)
    payload = client.chat_json(
        prompt,
        system_prompt=(
            "You generate one high-quality strategy idea. "
            "Stay faithful to the assigned persona. "
            "Return exactly one JSON object. "
            "Novelty matters more than realism. "
            "It is acceptable if the idea is unconventional, speculative, or partially unrealistic. "
            "Avoid filtering toward safe, ordinary, or over-practical answers. "
            "Do not collapse back into generic recommendation, assistant, dashboard, planner, or companion patterns "
            "unless that direction is clearly the most original and concrete option. "
            "The description should be concrete and moderately detailed. "
            "Explain mechanism, practical operation, and strategic difference. "
            f"Write all natural-language fields in {language}. "
            "Populate all requested fields with specific, usable content. "
            "Return only valid JSON."
        ),
        debug_label=f"generator#{index} model={model} persona={persona_name}",
        num_predict=GENERATOR_NUM_PREDICT,
    )
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
        "id": str(payload.get("id") or f"idea_{index}"),
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
