"""Creativity scoring for generated ideas using LLM-as-a-Judge."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from config import JUDGE_MODELS
from llm.ollama_client import AsyncOllamaClient


async def score_ideas(problem: str, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate ideas concurrently using an LLM-as-a-Judge."""
    if not ideas:
        return []

    print(f"[scoring] Starting scoring for {len(ideas)} ideas...")
    started_at = time.perf_counter()
    
    selected_model = random.choice(JUDGE_MODELS)
    client = AsyncOllamaClient(model=selected_model)
    
    tasks = [_score_single(problem, idea, client) for idea in ideas]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored_ideas: list[dict[str, Any]] = []
    for idea, res in zip(ideas, results):
        scored_idea = dict(idea)
        if isinstance(res, Exception) or res is None:
            print(f"[scoring] Failed to score {idea.get('id')}: {res}")
            scored_idea["scores"] = {"novelty": 5.0, "problem_fit": 5.0, "feasibility": 5.0}
        else:
            scored_idea["scores"] = res
            
        scored_idea["score_meta"] = {
            "method": "llm_as_a_judge",
            "model": selected_model,
        }
        scored_ideas.append(scored_idea)

    elapsed = time.perf_counter() - started_at
    print(f"[scoring] Scored {len(ideas)} ideas via {selected_model} in {elapsed:.1f}s (avg {elapsed/len(ideas):.1f}s/idea)")
    return scored_ideas


async def _score_single(problem: str, idea: dict[str, Any], client: AsyncOllamaClient) -> dict[str, float]:
    prompt = f"""
Problem: {problem}
Idea: {idea.get('title')}
Description: {idea.get('description')}
Mechanism: {idea.get('mechanism')}

Evaluate the idea strictly on a scale of 1.0 to 10.0 for the following criteria:
1. novelty: How radically different is this from standard/existing solutions? (1=Cliché, 10=Completely orthogonal paradigm)
2. problem_fit: How directly does this solve the root cause of the problem? (1=Irrelevant, 10=Perfect strike)
3. feasibility: How practically executable is this without magic/future technology? (1=Impossible, 10=Deployable tomorrow)

Output ONLY a JSON object with keys "novelty", "problem_fit", "feasibility" containing floating point numbers.
"""
    payload = await client.chat_json(
        user_prompt=prompt,
        system_prompt="You are a harsh but fair judge. Return exactly one JSON object.",
        debug_label=f"scoring_{idea.get('id')}",
        num_predict=1024
    )
    if not payload:
        return {"novelty": 5.0, "problem_fit": 5.0, "feasibility": 5.0}
        
    try:
        return {
            "novelty": float(payload.get("novelty", 5.0)),
            "problem_fit": float(payload.get("problem_fit", 5.0)),
            "feasibility": float(payload.get("feasibility", 5.0))
        }
    except (ValueError, TypeError):
        return {"novelty": 5.0, "problem_fit": 5.0, "feasibility": 5.0}
