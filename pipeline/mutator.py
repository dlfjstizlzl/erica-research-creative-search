"""Idea mutation (perturbation/refinement)."""

from __future__ import annotations

import asyncio
import random
import time
import uuid

from config import (
    DEFAULT_OUTPUT_LANGUAGE,
    MUTATION_COUNT,
    MUTATOR_ENFORCE_FAMILY,
    MUTATION_PROMPT,
    MUTATOR_NUM_PREDICT,
    MUTATOR_MODELS,
)
from core.utils import load_text
from llm.ollama_client import AsyncOllamaClient


async def mutate_idea(
    problem: str | dict,
    idea: dict | None = None,
    model: str | None = None,
    mutation_count: int = MUTATION_COUNT,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict]:
    """Generate multiple mutated variations of a parent idea asynchronously."""
    if not idea:
        print("[mutator] No idea provided for mutation.")
        return []

    parent_id = str(idea.get("id") or "")
    if isinstance(problem, dict):
        problem_text = str(problem.get("text") or problem.get("reframed") or "")
    else:
        problem_text = str(problem)

    prompt_template = load_text(MUTATION_PROMPT)
    selected_model = model or random.choice(MUTATOR_MODELS)
    client = AsyncOllamaClient(model=selected_model)

    tasks = [
        _mutate_single(
            problem_text=problem_text,
            idea=idea,
            language=language,
            prompt_template=prompt_template,
            client=client,
            index=index,
            total=mutation_count,
        )
        for index in range(1, mutation_count + 1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    mutated_ideas = []
    for res in results:
        if isinstance(res, Exception):
            print(f"[mutator] Mutation request failed for {parent_id}: {res}")
        elif res is not None:
            mutated_ideas.append(res)
            
    return mutated_ideas


async def _mutate_single(
    *,
    problem_text: str,
    idea: dict,
    language: str,
    prompt_template: str,
    client: AsyncOllamaClient,
    index: int,
    total: int,
) -> dict | None:
    parent_id = str(idea.get("id") or "")
    print(f"[mutator] Parent {parent_id}: generating variation {index}/{total}")
    started_at = time.perf_counter()

    prompt = prompt_template.format(
        problem=problem_text,
        id=parent_id,
        title=str(idea.get("title") or ""),
        strategy_type=str(idea.get("strategy_type") or ""),
        description=str(idea.get("description") or ""),
        mechanism=str(idea.get("mechanism") or ""),
        language=language,
        mutation_count=total,
    )

    payload = await client.chat_json(
        user_prompt=prompt,
        system_prompt=(
            "You are a mutator that takes an existing idea and twists it into a distinct variation. "
            "Return exactly one JSON object. "
            "Identify the core mechanism of the parent, then change its target audience, delivery channel, timescale, or underlying incentive structure. "
            "Do not output a completely unrelated idea. Do not output a slightly better copy. "
            "Make exactly one significant structural pivot. "
            "Do not make the idea significantly more generic or safe. "
            "Keep the result concrete and actionable. "
            f"Write all natural-language fields in {language}. "
            "Return only valid JSON."
        ),
        debug_label=f"mutator_{parent_id}_{index}",
        num_predict=MUTATOR_NUM_PREDICT,
    )
    if not payload:
        return None

    elapsed = time.perf_counter() - started_at
    mutated = _normalize(payload, idea, index)
    
    if mutated["description"]:
        print(f"[mutator] Parent {parent_id}: variation {index} ready in {elapsed:.1f}s -> {mutated['title']}")
        return mutated
        
    print(f"[mutator] Parent {parent_id}: variation {index} returned empty idea in {elapsed:.1f}s.")
    return None


def _normalize(payload: object, parent_idea: dict, index: int) -> dict:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise RuntimeError("Expected JSON object from mutation")

    parent_id = str(parent_idea.get("id") or "")
    new_id = uuid.uuid4().hex[:8]

    parent_title = str(parent_idea.get("title") or "")
    new_title = str(payload.get("title") or f"Mutated {parent_title} {index}").strip()

    parent_family = parent_idea.get("parent_ids") or [parent_id]
    if isinstance(parent_family, list) and parent_id not in parent_family:
        parent_family = list(parent_family) + [parent_id]

    depth = int(parent_idea.get("depth", 0)) + 1
    mutation_type = str(payload.get("mutation_type") or "general")

    strategy_type = str(payload.get("strategy_type") or parent_idea.get("strategy_type") or "general")
    if MUTATOR_ENFORCE_FAMILY:
        strategy_type = str(parent_idea.get("strategy_type") or strategy_type)

    return {
        "id": new_id,
        "title": new_title,
        "strategy_type": strategy_type,
        "description": " ".join(str(payload.get("description") or "").split()),
        "mechanism": " ".join(str(payload.get("mechanism") or "").split()),
        "target_user": " ".join(str(payload.get("target_user") or "").split()),
        "execution_context": " ".join(str(payload.get("execution_context") or "").split()),
        "expected_advantage": " ".join(str(payload.get("expected_advantage") or "").split()),
        "parent_id": parent_id,
        "parent_ids": parent_family,
        "depth": depth,
        "mutation_type": mutation_type,
        "origin_type": "mutation",
    }
