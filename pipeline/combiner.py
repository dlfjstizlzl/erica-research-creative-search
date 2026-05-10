"""Idea recombination for iterative search."""

from __future__ import annotations

from typing import Any

from config import COMBINATION_PROMPT, COMBINER_MODEL, DEFAULT_OUTPUT_LANGUAGE
from core.utils import load_text
from llm import OllamaClient


COMBINATION_TYPES = [
    "mechanism_plus_audience",
    "mechanism_plus_channel",
    "delivery_model_swap",
    "context_shift",
    "hybrid_system",
]

CONFLICT_FRAMES = [
    "individual incentive vs collective coordination",
    "automation efficiency vs human participation",
    "centralized orchestration vs local autonomy",
    "short-term adoption vs long-term behavior change",
    "measurement/control vs emotional or social engagement",
]


def combine_ideas(
    problem: str,
    parent_pairs: list[tuple[dict, dict]],
    *,
    model: str = COMBINER_MODEL,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
    generation: int = 1,
) -> list[dict]:
    """Combine parent pairs into new candidate ideas."""
    if not parent_pairs:
        return []

    prompt_template = load_text(COMBINATION_PROMPT)
    client = OllamaClient(model=model)
    combined: list[dict] = []
    print(
        f"[combiner] Starting recombination for {len(parent_pairs)} pairs "
        f"with model={model} generation={generation}"
    )

    for index, (left, right) in enumerate(parent_pairs, start=1):
        combination_type = COMBINATION_TYPES[(index - 1) % len(COMBINATION_TYPES)]
        conflict_frame = _infer_conflict_frame(left, right, index=index)
        left_id = str(left.get("id") or "")
        right_id = str(right.get("id") or "")
        print(
            f"[combiner] Pair {index}/{len(parent_pairs)} "
            f"parents={left_id}+{right_id} type={combination_type} "
            f"conflict={conflict_frame}"
        )
        algorithmic_draft = _create_algorithmic_draft(left, right, combination_type)
        
        prompt = prompt_template.format(
            problem=problem,
            algorithmic_draft=algorithmic_draft,
            combination_type=combination_type,
            language=language,
        )
        payload = client.chat_json(
            prompt,
            system_prompt=(
                "You are an expert editor and creative strategist. "
                "You will receive an 'Algorithmic Draft' that structurally combines two ideas. "
                "Your task is to refine and polish this draft into a natural, cohesive, and highly creative single system. "
                "Do not invent entirely new core mechanisms; stick to the structural skeleton provided in the draft. "
                "Focus on making the context natural, the mechanism clear, and the overall description compelling. "
                "Return exactly one JSON object. "
                "Return only valid JSON."
            ),
            debug_label=(
                f"combiner#{index} model={model} "
                f"parents={left.get('id', '')}+{right.get('id', '')}"
            ),
        )
        combined.extend(
            normalize_combination_output(
                payload,
                parent_ideas=[left, right],
                index=index,
                generation=generation,
                combination_type=combination_type,
                model=model,
            )
        )

    print(f"[combiner] Produced {len(combined)} combined ideas.")
    return combined


def _create_algorithmic_draft(left: dict, right: dict, combination_type: str) -> str:
    """Create a structural draft of the combined idea using pure logic."""
    draft = [f"[Combination Type: {combination_type}]"]
    
    if combination_type == "mechanism_plus_audience":
        draft.append(f"Core Mechanism: {left.get('mechanism', '')}")
        draft.append(f"Target Audience: {right.get('target_user', '')}")
        draft.append(f"Execution Context: {right.get('execution_context', '')}")
    elif combination_type == "mechanism_plus_channel":
        draft.append(f"Core Mechanism: {left.get('mechanism', '')}")
        draft.append(f"Delivery/Channel: {right.get('execution_context', '')}")
        draft.append(f"Target Audience: {left.get('target_user', '')}")
    elif combination_type == "delivery_model_swap":
        draft.append(f"Core Strategy: {left.get('strategy_type', '')}")
        draft.append(f"Execution Context: {right.get('execution_context', '')}")
        draft.append(f"Mechanism: {right.get('mechanism', '')}")
    elif combination_type == "context_shift":
        draft.append(f"Core Mechanism: {left.get('mechanism', '')}")
        draft.append(f"New Context: {right.get('execution_context', '')}")
    else:  # hybrid_system
        draft.append(f"Combined Mechanism: {left.get('mechanism', '')} AND {right.get('mechanism', '')}")
        draft.append(f"Combined Context: {left.get('execution_context', '')} AND {right.get('execution_context', '')}")
        
    return "\n".join(draft)


def normalize_combination_output(
    raw_output: Any,
    *,
    parent_ideas: list[dict],
    index: int,
    generation: int,
    combination_type: str,
    model: str,
) -> list[dict]:
    """Normalize combiner output into JSON-safe idea dicts."""
    if isinstance(raw_output, list):
        raw_output = raw_output[0] if raw_output else {}
    if not isinstance(raw_output, dict):
        return []

    left = parent_ideas[0]
    right = parent_ideas[1]
    parent_ids = [
        str(left.get("id") or "").strip(),
        str(right.get("id") or "").strip(),
    ]
    parent_ids = [parent_id for parent_id in parent_ids if parent_id]
    depth = max(int(left.get("depth") or 0), int(right.get("depth") or 0)) + 1

    combined = {
        "id": str(raw_output.get("id") or f"combo_{generation}_{index}"),
        "title": str(raw_output.get("title") or f"Combined Idea {index}").strip(),
        "strategy_type": str(raw_output.get("strategy_type") or "general").strip(),
        "description": " ".join(str(raw_output.get("description") or "").split()),
        "mechanism": " ".join(str(raw_output.get("mechanism") or "").split()),
        "target_user": " ".join(str(raw_output.get("target_user") or "").split()),
        "execution_context": " ".join(
            str(raw_output.get("execution_context") or "").split()
        ),
        "expected_advantage": " ".join(
            str(raw_output.get("expected_advantage") or "").split()
        ),
        "parent_ids": parent_ids,
        "depth": depth,
        "origin_type": "combination",
        "generation": generation,
        "combination_type": str(
            raw_output.get("combination_type") or combination_type
        ).strip(),
        "source_model": model.strip(),
        "source_persona": "combiner",
    }
    if not combined["title"] or not combined["description"]:
        print(
            f"[combiner] Dropped combined candidate for parents={'+'.join(parent_ids)} "
            "because title or description was empty."
        )
        return []
    if _is_shallow_combination(combined, left, right):
        print(
            f"[combiner] Dropped shallow combined idea {combined['id']} "
            f"from parents={'+'.join(parent_ids)}"
        )
        return []
    print(
        f"[combiner] Accepted combined idea {combined['id']} "
        f"from parents={'+'.join(parent_ids)}"
    )
    return [combined]


def _infer_conflict_frame(left: dict, right: dict, *, index: int) -> str:
    left_text = " ".join(
        [
            str(left.get("strategy_type") or ""),
            str(left.get("description") or ""),
            str(left.get("mechanism") or ""),
        ]
    ).lower()
    right_text = " ".join(
        [
            str(right.get("strategy_type") or ""),
            str(right.get("description") or ""),
            str(right.get("mechanism") or ""),
        ]
    ).lower()

    conflict_rules = [
        (
            {"game", "reward", "points", "badge", "incentive"},
            {"community", "collective", "sharing", "network", "peer"},
            "individual incentive vs collective coordination",
        ),
        (
            {"ai", "automation", "algorithm", "optimiz", "prediction"},
            {"community", "human", "peer", "ritual", "participation"},
            "automation efficiency vs human participation",
        ),
        (
            {"platform", "orchestr", "central", "dashboard"},
            {"local", "peer", "distributed", "neighborhood", "community"},
            "centralized orchestration vs local autonomy",
        ),
        (
            {"real-time", "instant", "immediate", "short-term"},
            {"habit", "ritual", "long-term", "culture", "education"},
            "short-term adoption vs long-term behavior change",
        ),
        (
            {"tracking", "measurement", "feedback", "sensor", "monitor"},
            {"emotion", "story", "music", "social", "narrative"},
            "measurement/control vs emotional or social engagement",
        ),
    ]

    for left_terms, right_terms, label in conflict_rules:
        if _contains_any(left_text, left_terms) and _contains_any(right_text, right_terms):
            return label
        if _contains_any(left_text, right_terms) and _contains_any(right_text, left_terms):
            return label

    return CONFLICT_FRAMES[(index - 1) % len(CONFLICT_FRAMES)]


def _is_shallow_combination(candidate: dict, left: dict, right: dict) -> bool:
    candidate_title = _normalize(candidate.get("title", ""))
    left_title = _normalize(left.get("title", ""))
    right_title = _normalize(right.get("title", ""))
    candidate_desc = _normalize(candidate.get("description", ""))
    left_desc = _normalize(left.get("description", ""))
    right_desc = _normalize(right.get("description", ""))
    candidate_mech = _normalize(candidate.get("mechanism", ""))
    left_mech = _normalize(left.get("mechanism", ""))
    right_mech = _normalize(right.get("mechanism", ""))

    if candidate_title and (candidate_title == left_title or candidate_title == right_title):
        return True
    if _ngram_jaccard(candidate_title, left_title) > 0.90:
        return True
    if _ngram_jaccard(candidate_title, right_title) > 0.90:
        return True
    if _ngram_jaccard(candidate_desc, left_desc) > 0.86 and _ngram_jaccard(candidate_mech, left_mech) > 0.72:
        return True
    if _ngram_jaccard(candidate_desc, right_desc) > 0.86 and _ngram_jaccard(candidate_mech, right_mech) > 0.72:
        return True
    return False


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())


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
