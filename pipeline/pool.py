"""Pool management for iterative search."""

from __future__ import annotations

from collections import defaultdict

from config import POOL_MAX_PER_STRATEGY, POOL_MAX_SIZE


def initialize_pool(ideas: list[dict], max_size: int = POOL_MAX_SIZE) -> list[dict]:
    """Create the initial active pool from scored seed ideas."""
    pool = preserve_diversity(ideas, max_size=max_size)
    print(
        f"[pool] Initialized active pool with {len(pool)}/{len(ideas)} ideas: "
        + ", ".join(str(idea.get("id") or "") for idea in pool)
    )
    return pool


def update_pool(
    existing_pool: list[dict],
    new_candidates: list[dict],
    *,
    max_size: int = POOL_MAX_SIZE,
) -> list[dict]:
    """Merge new candidates into the active pool and keep a diverse frontier."""
    updated = preserve_diversity([*existing_pool, *new_candidates], max_size=max_size)
    print(
        f"[pool] Updated pool from {len(existing_pool)} + {len(new_candidates)} "
        f"to {len(updated)} active ideas: "
        + ", ".join(str(idea.get("id") or "") for idea in updated)
    )
    return updated


def preserve_diversity(
    ideas: list[dict],
    *,
    max_size: int = POOL_MAX_SIZE,
    max_per_strategy: int = POOL_MAX_PER_STRATEGY,
) -> list[dict]:
    """Keep a small but diverse active pool."""
    ordered = sorted(ideas, key=_pool_priority, reverse=True)
    selected: list[dict] = []
    strategy_counts: dict[str, int] = defaultdict(int)
    signature_set: set[str] = set()

    for idea in ordered:
        signature = _idea_signature(idea)
        if signature in signature_set:
            continue

        strategy = _normalize(idea.get("strategy_type") or "general")
        if strategy_counts[strategy] >= max_per_strategy:
            continue

        selected.append(idea)
        signature_set.add(signature)
        strategy_counts[strategy] += 1

        if len(selected) >= max_size:
            break

    print(
        f"[pool] Diversity preservation kept {len(selected)}/{len(ideas)} ideas "
        f"(max_size={max_size}, max_per_strategy={max_per_strategy})"
    )
    return selected


def _pool_priority(idea: dict) -> tuple[float, float, float, int]:
    scores = idea.get("scores") or {}
    creativity = float(scores.get("creativity", 0.0))
    novelty = float(scores.get("novelty", 0.0))
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    description_len = len(str(idea.get("description") or "").split())
    return creativity, novelty, problem_fit, description_len


def _idea_signature(idea: dict) -> str:
    title = _normalize(idea.get("title") or "")
    strategy = _normalize(idea.get("strategy_type") or "")
    description = _normalize(idea.get("description") or "")
    origin = _normalize(idea.get("origin_type") or "")
    return " || ".join([title, strategy, description, origin])


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())
