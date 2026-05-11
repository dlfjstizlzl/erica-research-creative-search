"""Idea combination (dialectical synthesis)."""

from __future__ import annotations

import asyncio
import time
import random
import uuid
from typing import Any

from config import COMBINATION_PROMPT, COMBINER_MODELS, EXTRACTOR_MODELS, DEFAULT_OUTPUT_LANGUAGE
from core.utils import load_text
from llm.ollama_client import AsyncOllamaClient

COMBINATION_TYPES = [
    "dialectical_synthesis",
    "mechanism_transfer",
    "hybrid_system",
]


async def combine_ideas(
    problem: str | dict,
    parent_pairs: list[tuple[dict, dict]],
    model: str | None = None,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict]:
    """Generate combinations for multiple pairs of parent ideas asynchronously."""
    if not parent_pairs:
        return []

    if isinstance(problem, dict):
        problem_text = str(problem.get("text") or problem.get("reframed") or "")
    else:
        problem_text = str(problem)

    prompt_template = load_text(COMBINATION_PROMPT)
    
    # Select random models if not specified
    selected_model = model or random.choice(COMBINER_MODELS)
    client = AsyncOllamaClient(model=selected_model)
    extractor_client = AsyncOllamaClient(model=random.choice(EXTRACTOR_MODELS))

    tasks = [
        _combine_single(
            problem_text=problem_text,
            left=left,
            right=right,
            language=language,
            prompt_template=prompt_template,
            client=client,
            extractor_client=extractor_client,
            index=index,
        )
        for index, (left, right) in enumerate(parent_pairs, start=1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    combined_ideas = []
    for res in results:
        if isinstance(res, Exception):
            print(f"[combiner] Combination request failed: {res}")
        elif res is not None:
            combined_ideas.extend(res)
            
    return combined_ideas


async def _combine_single(
    *,
    problem_text: str,
    left: dict,
    right: dict,
    language: str,
    prompt_template: str,
    client: AsyncOllamaClient,
    extractor_client: AsyncOllamaClient,
    index: int,
) -> list[dict] | None:
    combination_type = COMBINATION_TYPES[(index - 1) % len(COMBINATION_TYPES)]
    
    # 1. Extract dynamic conflict
    conflict_frame = await _extract_dynamic_conflict(left, right, client=extractor_client, language=language)
    
    left_id = str(left.get("id") or "")
    right_id = str(right.get("id") or "")
    print(f"[combiner] Pair {left_id} x {right_id}: resolving tension [{conflict_frame}]")
    started_at = time.perf_counter()

    prompt = prompt_template.format(
        problem=problem_text,
        left_id=left_id,
        left_title=str(left.get("title") or ""),
        left_strategy_type=str(left.get("strategy_type") or ""),
        left_description=str(left.get("description") or ""),
        left_mechanism=str(left.get("mechanism") or ""),
        right_id=right_id,
        right_title=str(right.get("title") or ""),
        right_strategy_type=str(right.get("strategy_type") or ""),
        right_description=str(right.get("description") or ""),
        right_mechanism=str(right.get("mechanism") or ""),
        combination_type=combination_type,
        conflict_frame=conflict_frame,
        language=language,
    )

    payload = await client.chat_json(
        user_prompt=prompt,
        system_prompt=(
            "You are a strategic system architect. "
            "Return exactly one JSON object. "
            "Do not merge ideas by simply listing both. You must create a new synthesis that resolves the core tension. "
            "The result must feel like a genuinely new operating model. "
            f"Write all natural-language fields in {language}. "
            "Return only valid JSON."
        ),
        debug_label=f"combiner_{left_id}_{right_id}",
    )
    if not payload:
        return None

    elapsed = time.perf_counter() - started_at
    combined = _normalize(payload, left, right, combination_type)
    
    if _is_shallow_combination(combined, left, right):
        print(f"[combiner] Rejected shallow combination for pair {left_id} x {right_id}.")
        return []

    if combined["description"]:
        print(f"[combiner] Pair {left_id} x {right_id}: combination ready in {elapsed:.1f}s -> {combined['title']}")
        return [combined]
        
    print(f"[combiner] Pair {left_id} x {right_id}: combination returned empty idea in {elapsed:.1f}s.")
    return []


async def _extract_dynamic_conflict(left: dict, right: dict, *, client: AsyncOllamaClient, language: str) -> str:
    """Extract dynamic conflict frame using the LLM."""
    left_desc = str(left.get("description", ""))
    left_mech = str(left.get("mechanism", ""))
    left_user = str(left.get("target_user", ""))
    right_desc = str(right.get("description", ""))
    right_mech = str(right.get("mechanism", ""))
    right_user = str(right.get("target_user", ""))
    
    prompt = f"""
Idea A: {left_desc} / {left_mech} (Focus: {left_user})
Idea B: {right_desc} / {right_mech} (Focus: {right_user})

Identify the most fundamental 'strategic contradiction' or 'orthogonal tension' between these two ideas in a single short sentence.
Output the conflict in {language}.
"""
    try:
        payload = await client.chat_json(
            user_prompt=prompt,
            system_prompt="Return exactly one JSON object with a single key 'conflict' containing the 1-sentence contradiction. Return only valid JSON.",
            debug_label="combiner_conflict_extractor",
            num_predict=500,
        )
        conflict = str(payload.get("conflict", "")).strip()
        if conflict:
            return conflict
        return "Unknown conflict"
    except Exception as exc:
        print(f"[combiner] Failed to extract dynamic conflict: {exc}")
        return "Unknown conflict"


def _is_shallow_combination(candidate: dict, left: dict, right: dict) -> bool:
    candidate_desc = candidate.get("description", "").lower()
    left_desc = left.get("description", "").lower()
    right_desc = right.get("description", "").lower()

    if (
        len(candidate_desc) > 50
        and candidate_desc in left_desc
        or candidate_desc in right_desc
    ):
        return True
    
    if candidate.get("title") == left.get("title") or candidate.get("title") == right.get("title"):
        return True

    return False


def _normalize(value: object, left: dict, right: dict, ctype: str) -> dict:
    if isinstance(value, list):
        value = value[0] if value else {}
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object from combination")

    new_id = uuid.uuid4().hex[:8]

    left_id = str(left.get("id") or "")
    right_id = str(right.get("id") or "")

    l_parents = left.get("parent_ids") or [left_id]
    r_parents = right.get("parent_ids") or [right_id]
    merged_parents = list(set(l_parents).union(r_parents))

    depth = max(int(left.get("depth", 0)), int(right.get("depth", 0))) + 1

    reported_ctype = str(value.get("combination_type") or "").strip()
    final_ctype = reported_ctype if reported_ctype else ctype

    return {
        "id": new_id,
        "title": str(value.get("title") or f"Combined {left_id}+{right_id}").strip(),
        "strategy_type": str(value.get("strategy_type") or "hybrid").strip(),
        "description": " ".join(str(value.get("description") or "").split()),
        "mechanism": " ".join(str(value.get("mechanism") or "").split()),
        "target_user": " ".join(str(value.get("target_user") or "").split()),
        "execution_context": " ".join(str(value.get("execution_context") or "").split()),
        "expected_advantage": " ".join(str(value.get("expected_advantage") or "").split()),
        "parent_id": f"{left_id}+{right_id}",
        "parent_ids": merged_parents,
        "depth": depth,
        "combination_type": final_ctype,
        "origin_type": "combination",
    }
