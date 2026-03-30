"""Lightweight diversity filtering."""

from __future__ import annotations


def filter_diverse_ideas(ideas: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the raw pool broad and remove only obvious duplicates."""
    if len(ideas) <= 1:
        return ideas

    filtered: list[dict[str, str]] = []
    seen_signatures: set[str] = set()

    for idea in ideas:
        signature = _idea_signature(idea)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        filtered.append(idea)

    print(f"[filter] Light filter kept {len(filtered)}/{len(ideas)} ideas.")
    return filtered


def _idea_signature(idea: dict[str, str]) -> str:
    title = _normalize(idea.get("title", ""))
    strategy_type = _normalize(idea.get("strategy_type", ""))
    description = _normalize(idea.get("description", ""))
    parent_id = _normalize(idea.get("parent_id", ""))
    return " || ".join([title, strategy_type, description, parent_id])


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())
