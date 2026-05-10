"""Creativity scoring for generated ideas."""

from __future__ import annotations

from typing import Any


def score_ideas(problem: str, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach dummy novelty/relevance/creativity scores to idea dicts."""
    if not ideas:
        return []

    scored_ideas: list[dict[str, Any]] = []
    for idea in ideas:
        scored_idea = dict(idea)
        scored_idea["scores"] = {
            "novelty": 0.5,
            "problem_fit": 0.5,
            "relevance": 0.5,
            "mechanism_clarity": 0.5,
            "mutation_distance": 0.5,
            "mutation_quality": 0.5,
            "combination_quality": 0.5,
            "feasibility": 0.5,
            "risk": 0.5,
            "creativity": 0.5,
        }
        scored_idea["score_meta"] = {
            "method": "dummy",
            "version": "v5_dummy",
            "neighbor_count": 0,
        }
        scored_ideas.append(scored_idea)

    print(f"[scoring] Attached dummy scores (0.5) to {len(ideas)} ideas.")
    return scored_ideas
