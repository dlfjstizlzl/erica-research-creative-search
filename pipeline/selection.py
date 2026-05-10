"""Selection helpers for iterative search."""

from __future__ import annotations

from config import COMBINATION_PAIR_COUNT, PARENT_SELECTION_COUNT


def select_parent_ideas(
    pool_ideas: list[dict],
    *,
    exploit_count: int | None = None,
    explore_count: int = 1,
) -> list[dict]:
    """Choose a mix of high-scoring and high-novelty parents."""
    if not pool_ideas:
        return []

    exploit_count = exploit_count or max(1, PARENT_SELECTION_COUNT - explore_count)
    by_balanced = sorted(pool_ideas, key=_balanced_rank, reverse=True)
    by_wild = sorted(pool_ideas, key=_wild_rank, reverse=True)

    selected: list[dict] = []
    seen_ids: set[str] = set()

    for idea in by_balanced[:exploit_count]:
        idea_id = str(idea.get("id") or "")
        if idea_id and idea_id not in seen_ids:
            selected.append(idea)
            seen_ids.add(idea_id)

    for idea in by_wild[:explore_count]:
        idea_id = str(idea.get("id") or "")
        if idea_id and idea_id not in seen_ids:
            selected.append(idea)
            seen_ids.add(idea_id)

    exploit_ids = [str(idea.get("id") or "") for idea in by_balanced[:exploit_count]]
    explore_ids = [str(idea.get("id") or "") for idea in by_wild[:explore_count]]
    selected_ids = [str(idea.get("id") or "") for idea in selected]
    print(
        f"[selection] Parent selection "
        f"exploit={exploit_ids} explore={explore_ids} selected={selected_ids}"
    )
    return selected


def select_combination_pairs(
    pool_ideas: list[dict],
    *,
    max_pairs: int = COMBINATION_PAIR_COUNT,
) -> list[tuple[dict, dict]]:
    """Choose diverse parent pairs for recombination."""
    if len(pool_ideas) < 2:
        return []

    ordered = sorted(pool_ideas, key=_balanced_rank, reverse=True)
    candidate_pairs: list[tuple[tuple[float, float, float], dict, dict]] = []
    used_pair_ids: set[tuple[str, str]] = set()

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            left_id = str(left.get("id") or "")
            right_id = str(right.get("id") or "")
            pair_key = tuple(sorted([left_id, right_id]))
            if not left_id or not right_id or pair_key in used_pair_ids:
                continue
            if _too_similar(left, right):
                continue
            used_pair_ids.add(pair_key)
            candidate_pairs.append((_pair_rank(left, right), left, right))

    pairs = [(left, right) for _, left, right in sorted(candidate_pairs, key=lambda item: item[0], reverse=True)[:max_pairs]]
    if pairs:
        print(
            "[selection] Combination pairs selected: "
            + ", ".join(
                f"{str(pair_left.get('id') or '')}+{str(pair_right.get('id') or '')}"
                for pair_left, pair_right in pairs
            )
        )
    else:
        print("[selection] No valid combination pairs selected.")
    return pairs


def select_final_bests(ideas: list[dict]) -> dict[str, dict | None]:
    """Pick the best practical, balanced, and wild candidates."""
    if not ideas:
        return {
            "best_practical": None,
            "best_balanced": None,
            "best_wild": None,
        }

    remaining = list(ideas)
    best_practical = _pick_best(remaining, _practical_rank)
    remaining = _remove_selected(remaining, best_practical)
    best_balanced = _pick_best(remaining, _balanced_rank) or best_practical
    remaining = _remove_selected(remaining, best_balanced)
    best_wild = _pick_best(remaining, _wild_rank) or best_balanced or best_practical
    result = {
        "best_practical": best_practical,
        "best_balanced": best_balanced,
        "best_wild": best_wild,
    }
    print(
        "[selection] Final bests "
        f"practical={str(result['best_practical'].get('id') or '')} "
        f"balanced={str(result['best_balanced'].get('id') or '')} "
        f"wild={str(result['best_wild'].get('id') or '')}"
    )
    return result


def select_deep_judge_candidates(ideas: list[dict], *, top_ratio: float = 0.2) -> list[dict]:
    """Select the top N% candidates using multidimensional weights for Deep Judge."""
    if not ideas:
        return []
    
    target_count = max(1, int(len(ideas) * top_ratio))
    ranked = sorted(ideas, key=_deep_judge_rank, reverse=True)
    selected = ranked[:target_count]
    
    selected_ids = [str(idea.get("id") or "") for idea in selected]
    print(
        f"[selection] Deep Judge candidates selected ({len(selected)} out of {len(ideas)}): "
        f"{selected_ids}"
    )
    return selected


def _deep_judge_rank(idea: dict) -> tuple[float, float, float]:
    scores = idea.get("scores") or {}
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    feasibility = float(scores.get("feasibility", 0.0))
    mechanism = float(scores.get("mechanism_clarity", 0.0))
    novelty = float(scores.get("novelty", 0.0))
    creativity = float(scores.get("creativity", 0.0))
    risk = float(scores.get("risk", 0.0))
    
    # 1. 고정 가중치를 가진 논리성 점수 (Logic Score)
    logic_score = problem_fit * 0.4 + feasibility * 0.3 + mechanism * 0.3
    
    # 2. 독창성 점수 (Originality Score)
    originality_score = novelty * 0.5 + creativity * 0.5
    
    # 3. 논리성에 비례하는 가변 독창성 가중치 계산 (예: 0.1 ~ 0.3)
    # 논리성이 낮더라도 기본적으로 0.1의 가중치를 가지며, 논리성이 높을수록 가중치가 증가함
    originality_weight = 0.1 + (logic_score * 0.2)
    logic_weight = 0.7
    
    final_score = (
        logic_score * logic_weight
        + originality_score * originality_weight
        - risk * 0.2
    )
    return final_score, logic_score, originality_score


def _practical_rank(idea: dict) -> tuple[float, float, float, float]:
    scores = idea.get("scores") or {}
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    feasibility = float(scores.get("feasibility", 0.0))
    mechanism = float(scores.get("mechanism_clarity", 0.0))
    risk = float(scores.get("risk", 0.0))
    creativity = float(scores.get("creativity", 0.0))
    final_score = (
        problem_fit * 0.46
        + feasibility * 0.30
        + mechanism * 0.16
        + creativity * 0.08
        - risk * 0.38
    )
    return final_score, problem_fit, feasibility, mechanism


def _balanced_rank(idea: dict) -> tuple[float, float, float, float]:
    scores = idea.get("scores") or {}
    creativity = float(scores.get("creativity", 0.0))
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    novelty = float(scores.get("novelty", 0.0))
    feasibility = float(scores.get("feasibility", 0.0))
    mechanism = float(scores.get("mechanism_clarity", 0.0))
    risk = float(scores.get("risk", 0.0))
    final_score = (
        creativity * 0.24
        + problem_fit * 0.32
        + novelty * 0.06
        + feasibility * 0.22
        + mechanism * 0.16
        - risk * 0.34
    )
    return final_score, problem_fit, creativity, feasibility


def _wild_rank(idea: dict) -> tuple[float, float, float, float]:
    scores = idea.get("scores") or {}
    novelty = float(scores.get("novelty", 0.0))
    creativity = float(scores.get("creativity", 0.0))
    mechanism = float(scores.get("mechanism_clarity", 0.0))
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    risk = float(scores.get("risk", 0.0))
    mutation_quality = float(scores.get("mutation_quality", 0.0))
    combination_quality = float(scores.get("combination_quality", 0.0))
    final_score = (
        novelty * 0.40
        + creativity * 0.24
        + max(mutation_quality, combination_quality) * 0.16
        + mechanism * 0.10
        + problem_fit * 0.10
        - risk * 0.12
    )
    return final_score, novelty, creativity, mechanism


def _pair_rank(left: dict, right: dict) -> tuple[float, float, float]:
    left_scores = left.get("scores") or {}
    right_scores = right.get("scores") or {}
    pair_quality = (_balanced_rank(left)[0] + _balanced_rank(right)[0]) / 2.0
    strategy_diversity = 1.0 if _normalize(left.get("strategy_type") or "") != _normalize(right.get("strategy_type") or "") else 0.2
    origin_diversity = 1.0 if _normalize(left.get("origin_type") or "") != _normalize(right.get("origin_type") or "") else 0.3
    risk_penalty = (float(left_scores.get("risk", 0.0)) + float(right_scores.get("risk", 0.0))) / 2.0
    return pair_quality + strategy_diversity * 0.35 + origin_diversity * 0.20 - risk_penalty * 0.18, strategy_diversity, origin_diversity


def _pick_best(ideas: list[dict], rank_fn) -> dict | None:
    if not ideas:
        return None
    return max(ideas, key=rank_fn)


def _remove_selected(ideas: list[dict], selected: dict | None) -> list[dict]:
    if not selected:
        return ideas
    selected_id = str(selected.get("id") or "")
    return [
        idea
        for idea in ideas
        if str(idea.get("id") or "") != selected_id
    ]


def _too_similar(left: dict, right: dict) -> bool:
    left_id = _normalize(left.get("id") or "")
    right_id = _normalize(right.get("id") or "")
    left_strategy = _normalize(left.get("strategy_type") or "")
    right_strategy = _normalize(right.get("strategy_type") or "")
    left_origin = _normalize(left.get("origin_type") or "")
    right_origin = _normalize(right.get("origin_type") or "")
    left_parents = _normalized_parent_ids(left)
    right_parents = _normalized_parent_ids(right)
    direct_lineage_overlap = (
        left_id in right_parents
        or right_id in left_parents
        or bool(left_parents & right_parents)
    )
    if direct_lineage_overlap:
        return True
    return left_strategy == right_strategy and left_origin == right_origin


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _normalized_parent_ids(idea: dict) -> set[str]:
    parent_ids = {
        _normalize(item)
        for item in list(idea.get("parent_ids") or [])
        if _normalize(item)
    }
    parent_id = _normalize(idea.get("parent_id") or "")
    if parent_id:
        parent_ids.add(parent_id)
    return parent_ids
