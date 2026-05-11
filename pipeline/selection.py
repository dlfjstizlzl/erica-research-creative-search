"""Selection helpers for iterative search."""

from __future__ import annotations

import random
from config import COMBINATION_PAIR_COUNT, PARENT_SELECTION_COUNT


def select_parent_ideas(
    pool_ideas: list[dict],
    *,
    selection_count: int = PARENT_SELECTION_COUNT,
) -> list[dict]:
    """Choose parents using Pareto Front optimization (Multi-objective)."""
    if not pool_ideas:
        return []

    # Assign objectives based on LLM-as-a-judge scores
    for idea in pool_ideas:
        scores = idea.get("scores", {})
        # Objectives: maximize novelty, problem_fit, feasibility
        idea["_objs"] = (
            float(scores.get("novelty", 5.0)),
            float(scores.get("problem_fit", 5.0)),
            float(scores.get("feasibility", 5.0)),
        )

    fronts = _fast_non_dominated_sort(pool_ideas)
    selected: list[dict] = []
    seen_ids: set[str] = set()

    for front in fronts:
        # Sort within front by novelty as tiebreaker to maintain diversity
        front.sort(key=lambda x: x["_objs"][0], reverse=True)
        
        for idea in front:
            if len(selected) >= selection_count:
                break
            idea_id = str(idea.get("id") or "")
            if idea_id and idea_id not in seen_ids:
                selected.append(idea)
                seen_ids.add(idea_id)
        if len(selected) >= selection_count:
            break

    selected_ids = [str(idea.get("id") or "") for idea in selected]
    print(f"[selection] Pareto Parent Selection: selected={selected_ids}")
    return selected


def select_combination_pairs(
    pool_ideas: list[dict],
    *,
    max_pairs: int = COMBINATION_PAIR_COUNT,
) -> list[tuple[dict, dict]]:
    """Choose diverse parent pairs for recombination."""
    if len(pool_ideas) < 2:
        return []

    # Use first pareto front as candidates
    fronts = _fast_non_dominated_sort(pool_ideas)
    top_candidates = fronts[0]
    if len(top_candidates) < 2 and len(fronts) > 1:
        top_candidates.extend(fronts[1])

    candidate_pairs: list[tuple[float, dict, dict]] = []
    used_pair_ids: set[tuple[str, str]] = set()

    for left_index, left in enumerate(top_candidates):
        for right in top_candidates[left_index + 1 :]:
            left_id = str(left.get("id") or "")
            right_id = str(right.get("id") or "")
            pair_key = tuple(sorted([left_id, right_id]))
            if not left_id or not right_id or pair_key in used_pair_ids:
                continue
            if _too_similar(left, right):
                continue
                
            used_pair_ids.add(pair_key)
            # Rank pairs by combined novelty + diversity penalty
            pair_score = left["_objs"][0] + right["_objs"][0]
            candidate_pairs.append((pair_score, left, right))

    pairs = [(left, right) for _, left, right in sorted(candidate_pairs, key=lambda item: item[0], reverse=True)[:max_pairs]]
    
    if pairs:
        print("[selection] Combination pairs selected: " + ", ".join(f"{l.get('id')}+{r.get('id')}" for l, r in pairs))
    else:
        print("[selection] No valid combination pairs selected.")
    return pairs


def select_final_bests(ideas: list[dict]) -> dict[str, dict | None]:
    """Pick the best practical and wild candidates based on pareto fronts."""
    if not ideas:
        return {"best_practical": None, "best_balanced": None, "best_wild": None}

    # Recalculate objs just in case
    for idea in ideas:
        scores = idea.get("scores", {})
        idea["_objs"] = (
            float(scores.get("novelty", 5.0)),
            float(scores.get("problem_fit", 5.0)),
            float(scores.get("feasibility", 5.0)),
        )

    fronts = _fast_non_dominated_sort(ideas)
    top_front = fronts[0]

    # Best practical: Highest sum of problem_fit + feasibility
    best_practical = max(top_front, key=lambda x: x["_objs"][1] + x["_objs"][2])
    
    # Best wild: Highest novelty
    best_wild = max(top_front, key=lambda x: x["_objs"][0])
    
    # Best balanced: Highest sum of all three
    best_balanced = max(top_front, key=lambda x: sum(x["_objs"]))

    return {
        "best_practical": best_practical,
        "best_balanced": best_balanced,
        "best_wild": best_wild,
    }


def _fast_non_dominated_sort(ideas: list[dict]) -> list[list[dict]]:
    """O(N^2) fast non-dominated sorting."""
    fronts: list[list[dict]] = [[]]
    domination_counts = {id(p): 0 for p in ideas}
    dominated_sets = {id(p): [] for p in ideas}

    for p in ideas:
        for q in ideas:
            if p is q:
                continue
            if _dominates(p, q):
                dominated_sets[id(p)].append(q)
            elif _dominates(q, p):
                domination_counts[id(p)] += 1

        if domination_counts[id(p)] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[id(p)]:
                domination_counts[id(q)] -= 1
                if domination_counts[id(q)] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _dominates(p: dict, q: dict) -> bool:
    """Returns True if p strictly dominates q in all objectives."""
    p_objs = p["_objs"]
    q_objs = q["_objs"]
    
    # p must be >= q in all objectives, and > q in at least one
    better_in_one = False
    for p_val, q_val in zip(p_objs, q_objs):
        if p_val < q_val:
            return False
        if p_val > q_val:
            better_in_one = True
    return better_in_one


def _too_similar(left: dict, right: dict) -> bool:
    left_id = str(left.get("id") or "")
    right_id = str(right.get("id") or "")
    left_parents = _normalized_parent_ids(left)
    right_parents = _normalized_parent_ids(right)
    
    direct_lineage_overlap = (
        left_id in right_parents
        or right_id in left_parents
        or bool(left_parents & right_parents)
    )
    if direct_lineage_overlap:
        return True
    return False


def _normalized_parent_ids(idea: dict) -> set[str]:
    parent_ids = {str(item) for item in list(idea.get("parent_ids") or []) if item}
    parent_id = str(idea.get("parent_id") or "")
    if parent_id:
        parent_ids.add(parent_id)
    return parent_ids
