"""Lightweight diversity filtering."""

from __future__ import annotations


def filter_diverse_ideas(ideas: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the raw pool broad while removing obvious and near duplicates."""
    if len(ideas) <= 1:
        return ideas

    ordered = sorted(ideas, key=_filter_priority, reverse=True)
    filtered: list[dict[str, str]] = []
    seen_signatures: set[str] = set()

    for idea in ordered:
        signature = _idea_signature(idea)
        if signature in seen_signatures:
            continue
        if any(_near_duplicate(idea, existing) for existing in filtered):
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
    parent_ids = ",".join(
        sorted(_normalize(parent_id) for parent_id in list(idea.get("parent_ids") or []))
    )
    return " || ".join([title, strategy_type, description, parent_id, parent_ids])


def _filter_priority(idea: dict[str, str]) -> tuple[float, float, float, int]:
    scores = idea.get("scores") or {}
    creativity = float(scores.get("creativity", 0.0))
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    novelty = float(scores.get("novelty", 0.0))
    description_len = len(_normalize(idea.get("description", "")).split())
    return creativity, problem_fit, novelty, description_len


def _near_duplicate(left: dict[str, str], right: dict[str, str]) -> bool:
    left_title = _normalize(left.get("title", ""))
    right_title = _normalize(right.get("title", ""))
    left_strategy = _normalize(left.get("strategy_type", ""))
    right_strategy = _normalize(right.get("strategy_type", ""))
    left_desc = _normalize(left.get("description", ""))
    right_desc = _normalize(right.get("description", ""))
    left_mechanism = _normalize(left.get("mechanism", ""))
    right_mechanism = _normalize(right.get("mechanism", ""))
    left_origin = _normalize(left.get("origin_type", ""))
    right_origin = _normalize(right.get("origin_type", ""))
    left_parents = _normalized_parent_ids(left)
    right_parents = _normalized_parent_ids(right)

    title_sim = _ngram_jaccard(left_title, right_title)
    desc_sim = _ngram_jaccard(left_desc, right_desc)
    mech_sim = _ngram_jaccard(left_mechanism, right_mechanism)
    same_parent = _normalize(left.get("parent_id", "")) and _normalize(left.get("parent_id", "")) == _normalize(right.get("parent_id", ""))
    same_strategy = left_strategy and left_strategy == right_strategy
    shared_lineage = bool(left_parents & right_parents) or bool(
        {
            _normalize(left.get("id", "")),
            _normalize(right.get("id", "")),
        }
        & (left_parents | right_parents)
    )

    if left_title and left_title == right_title:
        return True
    if title_sim > 0.92 and desc_sim > 0.78:
        return True
    if same_strategy and title_sim > 0.88 and desc_sim > 0.55:
        return True
    if same_parent and desc_sim > 0.72:
        return True
    if same_strategy and desc_sim > 0.84:
        return True
    if desc_sim > 0.88 and mech_sim > 0.80:
        return True
    if left_origin == "combination" and right_origin == "combination" and title_sim > 0.80:
        return True
    if shared_lineage and title_sim > 0.72 and (desc_sim > 0.46 or mech_sim > 0.54):
        return True
    if shared_lineage and same_strategy and desc_sim > 0.60:
        return True
    return False


def _ngram_jaccard(left_text: str, right_text: str, n: int = 3) -> float:
    left = _char_ngrams(left_text, n=n)
    right = _char_ngrams(right_text, n=n)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = "".join(_normalize(text).split())
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _normalized_parent_ids(idea: dict[str, str]) -> set[str]:
    parent_ids = {
        _normalize(item)
        for item in list(idea.get("parent_ids") or [])
        if _normalize(item)
    }
    parent_id = _normalize(idea.get("parent_id", ""))
    if parent_id:
        parent_ids.add(parent_id)
    return parent_ids
