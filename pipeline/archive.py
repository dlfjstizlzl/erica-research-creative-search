"""Archive utilities for iterative search."""

from __future__ import annotations


def initialize_archive(ideas: list[dict]) -> list[dict]:
    """Create initial archive records from seed ideas."""
    archive = [_record_for_idea(idea, survived=True) for idea in ideas]
    print(f"[archive] Initialized archive with {len(archive)} records.")
    return archive


def update_archive(
    archive: list[dict],
    ideas: list[dict],
    *,
    active_ids: set[str],
) -> list[dict]:
    """Append new records while avoiding duplicate idea ids."""
    existing_ids = {str(record.get("idea_id") or "") for record in archive}
    updated = list(archive)
    added = 0
    for idea in ideas:
        idea_id = str(idea.get("id") or "")
        if not idea_id or idea_id in existing_ids:
            continue
        updated.append(_record_for_idea(idea, survived=idea_id in active_ids))
        added += 1
    print(
        f"[archive] Added {added} new records. "
        f"Archive size is now {len(updated)}."
    )
    return updated


def mark_selection_in_archive(
    archive: list[dict],
    *,
    best_practical_id: str = "",
    best_balanced_id: str = "",
    best_wild_id: str = "",
    active_ids: set[str] | None = None,
) -> list[dict]:
    """Mark final selections and final survival state in archive records."""
    active_ids = active_ids or set()
    updated: list[dict] = []
    for record in archive:
        record = dict(record)
        idea_id = str(record.get("idea_id") or "")
        record["survived"] = idea_id in active_ids
        labels: list[str] = []
        if idea_id and idea_id == best_practical_id:
            labels.append("best_practical")
        if idea_id and idea_id == best_balanced_id:
            labels.append("best_balanced")
        if idea_id and idea_id == best_wild_id:
            labels.append("best_wild")
        if labels:
            record["selected_labels"] = labels
        updated.append(record)
    print(
        "[archive] Final selection markers applied "
        f"(practical={best_practical_id or '-'}, "
        f"balanced={best_balanced_id or '-'}, "
        f"wild={best_wild_id or '-'})"
    )
    return updated


def summarize_archive(archive: list[dict]) -> dict:
    """Build a compact archive summary for analysis."""
    by_origin: dict[str, int] = {}
    survived_by_origin: dict[str, int] = {}
    by_generation: dict[str, int] = {}
    selected_ids: dict[str, str] = {}

    for record in archive:
        origin = str(record.get("origin_type") or "unknown")
        generation = str(record.get("generation") if record.get("generation") is not None else 0)
        by_origin[origin] = by_origin.get(origin, 0) + 1
        by_generation[generation] = by_generation.get(generation, 0) + 1
        if record.get("survived"):
            survived_by_origin[origin] = survived_by_origin.get(origin, 0) + 1
        for label in list(record.get("selected_labels") or []):
            selected_ids[str(label)] = str(record.get("idea_id") or "")

    summary = {
        "total_records": len(archive),
        "by_origin": by_origin,
        "survived_by_origin": survived_by_origin,
        "by_generation": by_generation,
        "selected_ids": selected_ids,
    }
    print(
        "[archive] Summary "
        f"total={summary['total_records']} "
        f"by_origin={summary['by_origin']} "
        f"survived_by_origin={summary['survived_by_origin']}"
    )
    return summary


def _record_for_idea(idea: dict, *, survived: bool) -> dict:
    parent_ids = [
        str(item)
        for item in list(idea.get("parent_ids") or [])
        if str(item).strip()
    ]
    if not parent_ids and str(idea.get("parent_id") or "").strip():
        parent_ids = [str(idea.get("parent_id")).strip()]

    return {
        "idea_id": str(idea.get("id") or ""),
        "title": str(idea.get("title") or ""),
        "strategy_type": str(idea.get("strategy_type") or ""),
        "origin_type": str(idea.get("origin_type") or "unknown"),
        "generation": int(idea.get("generation") or 0),
        "parent_ids": parent_ids,
        "survived": survived,
        "scores": {
            str(key): float(value)
            for key, value in dict(idea.get("scores") or {}).items()
            if isinstance(value, (int, float))
        },
    }
