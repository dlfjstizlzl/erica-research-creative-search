"""Creativity scoring for generated ideas."""

from __future__ import annotations

import math
from typing import Any

from config import OLLAMA_EMBED_MODEL, SCORING_NEIGHBOR_COUNT
from llm import OllamaClient


def score_ideas(problem: str, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach novelty/relevance/creativity scores to idea dicts."""
    if not ideas:
        return []

    try:
        scored = _score_with_embeddings(problem, ideas)
        _log_score_summary("embedding", scored)
        return scored
    except Exception as exc:
        print(f"[scoring] Embedding-based scoring failed: {exc}")
        print("[scoring] Falling back to lexical heuristic scoring.")
        scored = _score_with_heuristics(problem, ideas)
        _log_score_summary("heuristic", scored)
        return scored


def _score_with_embeddings(problem: str, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [_idea_to_text(idea) for idea in ideas]
    query_texts = _build_problem_queries(problem)
    client = OllamaClient(model=OLLAMA_EMBED_MODEL)
    print(
        f"[scoring] Running embedding-based scoring with model={OLLAMA_EMBED_MODEL} "
        f"for {len(ideas)} ideas."
    )
    embeddings = client.embed([*query_texts, *texts])
    if len(embeddings) != len(query_texts) + len(ideas):
        raise RuntimeError("Embedding count does not match inputs.")

    query_embeddings = embeddings[: len(query_texts)]
    idea_embeddings = embeddings[len(query_texts) :]
    id_to_index = {
        str(idea.get("id") or ""): index
        for index, idea in enumerate(ideas)
    }
    neighbor_count = max(1, min(SCORING_NEIGHBOR_COUNT, len(ideas) - 1))

    density_values = [
        _average_nearest_similarity(index, idea_embeddings, neighbor_count)
        for index in range(len(idea_embeddings))
    ]
    novelty_core = _invert_and_scale(density_values, neutral=0.5)
    relevance_raw = [
        _aggregate_query_relevance(idea_embeddings[index], query_embeddings)
        for index in range(len(idea_embeddings))
    ]
    relevance_relative = _scale_values(relevance_raw, neutral=0.5)
    lexical_fit_values = [
        _lexical_problem_fit(problem, idea)
        for idea in ideas
    ]

    scored_ideas: list[dict[str, Any]] = []
    for index, idea in enumerate(ideas):
        novelty = _clamp01(
            novelty_core[index] * 0.85 + _rarity_bonus(idea, ideas) * 0.15
        )
        problem_fit = _clamp01(
            relevance_raw[index] * 0.45
            + relevance_relative[index] * 0.20
            + lexical_fit_values[index] * 0.35
        )
        mechanism_clarity = _mechanism_clarity(idea)
        feasibility = _feasibility(idea, mechanism_clarity)
        risk = _risk_score(idea)

        parent_index = id_to_index.get(str(idea.get("parent_id") or ""))
        mutation_distance = 0.0
        mutation_quality = 0.0
        combination_quality = 0.0
        if parent_index is not None:
            mutation_distance = 1.0 - _normalize_cosine(
                _cosine_similarity(idea_embeddings[index], idea_embeddings[parent_index])
            )
            mutation_quality = _mutation_quality(
                idea=idea,
                parent_idea=ideas[parent_index],
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
                mutation_distance=mutation_distance,
            )
        combination_parent_indices = [
            id_to_index[parent_id]
            for parent_id in list(idea.get("parent_ids") or [])
            if parent_id in id_to_index
        ]
        if len(combination_parent_indices) >= 2:
            combination_quality = _combination_quality(
                idea_embedding=idea_embeddings[index],
                parent_embeddings=[idea_embeddings[parent_index] for parent_index in combination_parent_indices],
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
            )

        creativity = _combine_scores(
            novelty=novelty,
            problem_fit=problem_fit,
            mechanism_clarity=mechanism_clarity,
            mutation_quality=mutation_quality,
            combination_quality=combination_quality,
            feasibility=feasibility,
            risk=risk,
            is_mutation=parent_index is not None,
            is_combination=len(combination_parent_indices) >= 2,
        )
        scored_ideas.append(
            _with_scores(
                idea,
                method="embedding",
                novelty=novelty,
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
                mutation_distance=mutation_distance,
                mutation_quality=mutation_quality,
                combination_quality=combination_quality,
                feasibility=feasibility,
                risk=risk,
                creativity=creativity,
            )
        )

    return scored_ideas


def _score_with_heuristics(problem: str, ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_tokens = [set(_tokenize(query)) for query in _build_problem_queries(problem)]
    neighbor_count = max(1, min(SCORING_NEIGHBOR_COUNT, len(ideas) - 1))
    density_values: list[float] = []
    problem_fit_raw: list[float] = []
    lexical_bodies = [
        set(_tokenize(_idea_to_text(idea)))
        for idea in ideas
    ]

    for index, idea_tokens in enumerate(lexical_bodies):
        similarities = [
            _jaccard(idea_tokens, other_tokens)
            for other_index, other_tokens in enumerate(lexical_bodies)
            if other_index != index
        ]
        nearest = sorted(similarities, reverse=True)[:neighbor_count]
        density_values.append(sum(nearest) / len(nearest) if nearest else 0.0)
        query_overlaps = [_jaccard(tokens, idea_tokens) for tokens in query_tokens]
        problem_fit_raw.append(
            _clamp01(
                sum(query_overlaps) / len(query_overlaps) * 0.4
                + _lexical_problem_fit(problem, ideas[index]) * 0.6
            )
        )

    novelty_core = _invert_and_scale(density_values, neutral=0.5)
    problem_fit_relative = _scale_values(problem_fit_raw, neutral=0.5)

    by_id = {
        str(idea.get("id") or ""): idea
        for idea in ideas
    }
    scored_ideas: list[dict[str, Any]] = []
    for index, idea in enumerate(ideas):
        problem_fit = _clamp01(
            problem_fit_raw[index] * 0.7
            + problem_fit_relative[index] * 0.3
        )
        novelty = _clamp01(
            novelty_core[index] * 0.8 + _rarity_bonus(idea, ideas) * 0.2
        )
        mechanism_clarity = _mechanism_clarity(idea)
        feasibility = _feasibility(idea, mechanism_clarity)
        risk = _risk_score(idea)

        mutation_distance = 0.0
        mutation_quality = 0.0
        combination_quality = 0.0
        parent = by_id.get(str(idea.get("parent_id") or ""))
        if parent:
            parent_text = f"{parent.get('title', '')} {parent.get('description', '')}".lower()
            mutation_distance = 1.0 - _ngram_jaccard(parent_text, _idea_to_text(idea))
            mutation_quality = _mutation_quality(
                idea=idea,
                parent_idea=parent,
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
                mutation_distance=mutation_distance,
            )
        combination_parents = [
            by_id[parent_id]
            for parent_id in list(idea.get("parent_ids") or [])
            if parent_id in by_id
        ]
        if len(combination_parents) >= 2:
            combination_quality = _heuristic_combination_quality(
                idea=idea,
                parent_ideas=combination_parents,
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
            )

        creativity = _combine_scores(
            novelty=novelty,
            problem_fit=problem_fit,
            mechanism_clarity=mechanism_clarity,
            mutation_quality=mutation_quality,
            combination_quality=combination_quality,
            feasibility=feasibility,
            risk=risk,
            is_mutation=parent is not None,
            is_combination=len(combination_parents) >= 2,
        )
        scored_ideas.append(
            _with_scores(
                idea,
                method="heuristic",
                novelty=novelty,
                problem_fit=problem_fit,
                mechanism_clarity=mechanism_clarity,
                mutation_distance=mutation_distance,
                mutation_quality=mutation_quality,
                combination_quality=combination_quality,
                feasibility=feasibility,
                risk=risk,
                creativity=creativity,
            )
        )

    return scored_ideas


def _average_nearest_similarity(
    index: int,
    embeddings: list[list[float]],
    neighbor_count: int,
) -> float:
    if len(embeddings) <= 1:
        return 0.0

    similarities = [
        _normalize_cosine(_cosine_similarity(embeddings[index], other))
        for other_index, other in enumerate(embeddings)
        if other_index != index
    ]
    nearest = sorted(similarities, reverse=True)[:neighbor_count]
    if not nearest:
        return 0.0
    return sum(nearest) / len(nearest)


def _combine_scores(
    *,
    novelty: float,
    problem_fit: float,
    mechanism_clarity: float,
    mutation_quality: float,
    combination_quality: float,
    feasibility: float,
    risk: float,
    is_mutation: bool,
    is_combination: bool,
) -> float:
    if is_combination:
        score = (
            novelty * 0.30
            + problem_fit * 0.25
            + combination_quality * 0.30
            + mechanism_clarity * 0.15
        )
    elif is_mutation:
        score = (
            novelty * 0.35
            + problem_fit * 0.30
            + mutation_quality * 0.25
            + mechanism_clarity * 0.10
        )
    else:
        score = (
            novelty * 0.55
            + problem_fit * 0.35
            + mechanism_clarity * 0.10
        )

    # Keep feasibility and risk as light-touch adjustments instead of hard
    # gates so the evaluator does not collapse output diversity.
    score += (feasibility - 0.5) * 0.06
    score -= risk * 0.08
    return _clamp01(score)


def _build_problem_queries(problem: str) -> list[str]:
    return [
        problem,
        f"Directly solve this problem with a concrete system: {problem}",
        f"Provide a strategically novel but relevant solution to this problem: {problem}",
    ]


def _idea_to_text(idea: dict[str, Any]) -> str:
    parts = [
        f"title: {str(idea.get('title') or '').strip()}",
        f"persona: {str(idea.get('persona') or '').strip()}",
        f"strategy_type: {str(idea.get('strategy_type') or '').strip()}",
        f"description: {str(idea.get('description') or '').strip()}",
        f"mechanism: {str(idea.get('mechanism') or '').strip()}",
        f"target_user: {str(idea.get('target_user') or '').strip()}",
        f"context: {str(idea.get('execution_context') or '').strip()}",
        f"advantage: {str(idea.get('expected_advantage') or '').strip()}",
    ]
    if idea.get("mutation_type"):
        parts.append(f"mutation_type: {str(idea.get('mutation_type')).strip()}")
    return "\n".join(part for part in parts if part.strip())


def _normalize_cosine(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _aggregate_query_relevance(
    idea_embedding: list[float],
    query_embeddings: list[list[float]],
) -> float:
    similarities = [
        _normalize_cosine(_cosine_similarity(idea_embedding, query_embedding))
        for query_embedding in query_embeddings
    ]
    average_similarity = sum(similarities) / len(similarities)
    best_similarity = max(similarities)
    return _clamp01(average_similarity * 0.7 + best_similarity * 0.3)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        raise RuntimeError("Embedding vectors must be non-empty and equal length.")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = "".join(text.lower().split())
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _ngram_jaccard(left_text: str, right_text: str, n: int = 3) -> float:
    left = _char_ngrams(left_text, n=n)
    right = _char_ngrams(right_text, n=n)
    return _jaccard(left, right)


def _scale_values(values: list[float], *, neutral: float) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [neutral for _ in values]
    return [(value - low) / (high - low) for value in values]


def _invert_and_scale(values: list[float], *, neutral: float) -> list[float]:
    scaled = _scale_values(values, neutral=neutral)
    return [1.0 - value for value in scaled]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rarity_bonus(idea: dict[str, Any], ideas: list[dict[str, Any]]) -> float:
    strategy = str(idea.get("strategy_type") or "").strip().lower()
    source_model = str(idea.get("source_model") or "").strip().lower()
    source_persona = str(idea.get("source_persona") or "").strip().lower()

    strategy_count = sum(
        1
        for other in ideas
        if str(other.get("strategy_type") or "").strip().lower() == strategy
    )
    model_count = sum(
        1
        for other in ideas
        if str(other.get("source_model") or "").strip().lower() == source_model
    )
    persona_count = sum(
        1
        for other in ideas
        if str(other.get("source_persona") or "").strip().lower() == source_persona
    )

    components: list[float] = []
    if strategy:
        components.append(1.0 / max(1, strategy_count))
    if source_model:
        components.append(1.0 / max(1, model_count))
    if source_persona:
        components.append(1.0 / max(1, persona_count))
    if not components:
        return 0.5
    return sum(components) / len(components)


def _lexical_problem_fit(problem: str, idea: dict[str, Any]) -> float:
    idea_text = _idea_to_text(idea)
    query_texts = _build_problem_queries(problem)
    similarities = [_ngram_jaccard(query_text, idea_text) for query_text in query_texts]
    strongest_similarity = max(similarities) if similarities else 0.0

    mechanism = str(idea.get("mechanism") or "").strip()
    target_user = str(idea.get("target_user") or "").strip()
    expected_advantage = str(idea.get("expected_advantage") or "").strip()
    structure_bonus = (
        (0.08 if mechanism else 0.0)
        + (0.04 if target_user else 0.0)
        + (0.04 if expected_advantage else 0.0)
    )
    return _clamp01(strongest_similarity * 0.84 + structure_bonus)


def _mechanism_clarity(idea: dict[str, Any]) -> float:
    description = str(idea.get("description") or "").strip()
    mechanism = str(idea.get("mechanism") or "").strip()
    target_user = str(idea.get("target_user") or "").strip()
    execution_context = str(idea.get("execution_context") or "").strip()
    expected_advantage = str(idea.get("expected_advantage") or "").strip()
    combined = f"{description} {mechanism}".lower()

    word_detail = min(len(description.split()) / 70.0, 1.0) * 0.22
    mechanism_detail = min(len(mechanism.split()) / 35.0, 1.0) * 0.30
    field_presence = (
        (0.14 if target_user else 0.0)
        + (0.14 if execution_context else 0.0)
        + (0.14 if expected_advantage else 0.0)
    )
    structure_markers = [
        "because",
        "through",
        "via",
        "using",
        "based on",
        "통해",
        "기반",
        "분석",
        "활용",
        "연동",
        "자동",
        "단계",
    ]
    structure_bonus = 0.20 if any(marker in combined for marker in structure_markers) else 0.05
    return _clamp01(word_detail + mechanism_detail + field_presence + structure_bonus)


def _feasibility(idea: dict[str, Any], mechanism_clarity: float) -> float:
    execution_context = str(idea.get("execution_context") or "").strip()
    target_user = str(idea.get("target_user") or "").strip()
    expected_advantage = str(idea.get("expected_advantage") or "").strip()
    text = _idea_to_text(idea).lower()

    concreteness = (
        mechanism_clarity * 0.55
        + (0.15 if execution_context else 0.0)
        + (0.15 if target_user else 0.0)
        + (0.15 if expected_advantage else 0.0)
    )
    speculative_keywords = [
        "telepathy",
        "soul",
        "spirit",
        "mind control",
        "time travel",
        "영혼",
        "염력",
        "초능력",
        "순간이동",
        "시간여행",
    ]
    speculative_hits = sum(1 for keyword in speculative_keywords if keyword in text)
    penalty = min(speculative_hits * 0.12, 0.35)
    return _clamp01(concreteness - penalty)


def _risk_score(idea: dict[str, Any]) -> float:
    text = _idea_to_text(idea).lower()
    risk_keywords = [
        "manipulation",
        "deception",
        "coercion",
        "exploit",
        "surveillance",
        "misinformation",
        "addiction",
        "조작",
        "기만",
        "속이",
        "유도",
        "감시",
        "허위",
        "중독",
        "세뇌",
        "침투",
    ]
    hits = sum(1 for keyword in risk_keywords if keyword in text)
    return _clamp01(hits * 0.18)


def _mutation_quality(
    *,
    idea: dict[str, Any],
    parent_idea: dict[str, Any],
    problem_fit: float,
    mechanism_clarity: float,
    mutation_distance: float,
) -> float:
    child_strategy = str(idea.get("strategy_type") or "").strip().lower()
    parent_strategy = str(parent_idea.get("strategy_type") or "").strip().lower()
    strategy_shift = 1.0 if child_strategy and child_strategy != parent_strategy else 0.35
    mutation_type = str(idea.get("mutation_type") or "").strip().lower()
    mutation_type_bonus = 1.0 if mutation_type and mutation_type not in {"variation", "same", "general"} else 0.5
    return _clamp01(
        mutation_distance * 0.40
        + problem_fit * 0.20
        + mechanism_clarity * 0.15
        + strategy_shift * 0.15
        + mutation_type_bonus * 0.10
    )


def _combination_quality(
    *,
    idea_embedding: list[float],
    parent_embeddings: list[list[float]],
    problem_fit: float,
    mechanism_clarity: float,
) -> float:
    parent_distances = [
        1.0 - _normalize_cosine(_cosine_similarity(idea_embedding, parent_embedding))
        for parent_embedding in parent_embeddings
    ]
    average_parent_distance = sum(parent_distances) / len(parent_distances)
    return _clamp01(
        average_parent_distance * 0.45
        + problem_fit * 0.25
        + mechanism_clarity * 0.20
        + 0.10
    )


def _heuristic_combination_quality(
    *,
    idea: dict[str, Any],
    parent_ideas: list[dict[str, Any]],
    problem_fit: float,
    mechanism_clarity: float,
) -> float:
    idea_text = _idea_to_text(idea).lower()
    parent_texts = [
        _idea_to_text(parent_idea).lower()
        for parent_idea in parent_ideas
    ]
    parent_distances = [
        1.0 - _ngram_jaccard(parent_text, idea_text)
        for parent_text in parent_texts
    ]
    average_parent_distance = sum(parent_distances) / len(parent_distances)
    return _clamp01(
        average_parent_distance * 0.45
        + problem_fit * 0.25
        + mechanism_clarity * 0.20
        + 0.10
    )


def _with_scores(
    idea: dict[str, Any],
    *,
    method: str,
    novelty: float,
    problem_fit: float,
    mechanism_clarity: float,
    mutation_distance: float,
    mutation_quality: float,
    combination_quality: float,
    feasibility: float,
    risk: float,
    creativity: float,
) -> dict[str, Any]:
    scored_idea = dict(idea)
    scored_idea["scores"] = {
        "novelty": round(novelty, 4),
        "problem_fit": round(problem_fit, 4),
        "relevance": round(problem_fit, 4),
        "mechanism_clarity": round(mechanism_clarity, 4),
        "mutation_distance": round(mutation_distance, 4),
        "mutation_quality": round(mutation_quality, 4),
        "combination_quality": round(combination_quality, 4),
        "feasibility": round(feasibility, 4),
        "risk": round(risk, 4),
        "creativity": round(creativity, 4),
    }
    scored_idea["score_meta"] = {
        "method": method,
        "version": "v3_balanced",
        "neighbor_count": SCORING_NEIGHBOR_COUNT,
    }
    return scored_idea


def _log_score_summary(method: str, ideas: list[dict[str, Any]]) -> None:
    if not ideas:
        print(f"[scoring] No ideas to score with method={method}.")
        return

    novelty_scores = []
    problem_fit_scores = []
    mechanism_scores = []
    mutation_quality_scores = []
    combination_quality_scores = []
    feasibility_scores = []
    risk_scores = []
    creativity_scores = []
    mutation_distance_scores = []
    for idea in ideas:
        scores = idea.get("scores") or {}
        novelty_scores.append(float(scores.get("novelty", 0.0)))
        problem_fit_scores.append(float(scores.get("problem_fit", scores.get("relevance", 0.0))))
        mechanism_scores.append(float(scores.get("mechanism_clarity", 0.0)))
        mutation_quality_scores.append(float(scores.get("mutation_quality", 0.0)))
        combination_quality_scores.append(float(scores.get("combination_quality", 0.0)))
        feasibility_scores.append(float(scores.get("feasibility", 0.0)))
        risk_scores.append(float(scores.get("risk", 0.0)))
        creativity_scores.append(float(scores.get("creativity", 0.0)))
        mutation_distance_scores.append(float(scores.get("mutation_distance", 0.0)))

    print(
        f"[scoring] method={method} "
        f"novelty=[{min(novelty_scores):.4f}, {max(novelty_scores):.4f}] "
        f"problem_fit=[{min(problem_fit_scores):.4f}, {max(problem_fit_scores):.4f}] "
        f"mechanism=[{min(mechanism_scores):.4f}, {max(mechanism_scores):.4f}] "
        f"mutation_quality=[{min(mutation_quality_scores):.4f}, {max(mutation_quality_scores):.4f}] "
        f"combination_quality=[{min(combination_quality_scores):.4f}, {max(combination_quality_scores):.4f}] "
        f"feasibility=[{min(feasibility_scores):.4f}, {max(feasibility_scores):.4f}] "
        f"risk=[{min(risk_scores):.4f}, {max(risk_scores):.4f}] "
        f"creativity=[{min(creativity_scores):.4f}, {max(creativity_scores):.4f}] "
        f"mutation_distance=[{min(mutation_distance_scores):.4f}, {max(mutation_distance_scores):.4f}]"
    )
