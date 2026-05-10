"""Idea recombination for iterative search."""

from __future__ import annotations

from typing import Any

from config import COMBINATION_PROMPT, COMBINER_MODEL, EXTRACTOR_MODEL, DEFAULT_OUTPUT_LANGUAGE
from core.utils import load_text
from llm import OllamaClient


COMBINATION_TYPES = [
    "mechanism_plus_audience",
    "mechanism_plus_channel",
    "delivery_model_swap",
    "context_shift",
    "hybrid_system",
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
        conflict_frame = _extract_dynamic_conflict(left, right, model=EXTRACTOR_MODEL, language=language)
        left_id = str(left.get("id") or "")
        right_id = str(right.get("id") or "")
        print(
            f"[combiner] Pair {index}/{len(parent_pairs)} "
            f"parents={left_id}+{right_id} type={combination_type} "
            f"conflict={conflict_frame}"
        )
        prompt = prompt_template.format(
            problem=problem,
            left_id=left_id,
            left_title=left.get("title", ""),
            left_strategy_type=left.get("strategy_type", ""),
            left_description=left.get("description", ""),
            left_mechanism=left.get("mechanism", ""),
            right_id=right_id,
            right_title=right.get("title", ""),
            right_strategy_type=right.get("strategy_type", ""),
            right_description=right.get("description", ""),
            right_mechanism=right.get("mechanism", ""),
            combination_type=combination_type,
            conflict_frame=conflict_frame,
            language=language,
        )
        payload = client.chat_json(
            prompt,
            system_prompt=(
                "You combine two ideas into one genuinely new system. "
                "Do not average them. "
                "Do not list both ideas side by side. "
                "First identify the strongest strategic conflict between the parents. "
                "Then create a new system that resolves, exploits, or reframes that conflict. "
                "Use one concrete mechanism from the left parent and one concrete "
                "delivery, audience, context, or interaction element from the right parent. "
                "Reject shallow hybrids and feature bundles. "
                "Do not simply rename a parent idea or keep the same naming stem. "
                "Do not produce a direct sequel, polished variant, or obvious merge of nearly identical parents. "
                "The result must feel like a new operating model born from tension. "
                "Maintain clear problem-fit to the original problem. "
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


def _extract_dynamic_conflict(left: dict, right: dict, *, model: str, language: str) -> str:
    """Extract dynamic conflict frame using the LLM."""
    client = OllamaClient(model=model)
    left_desc = str(left.get("description", ""))
    left_mech = str(left.get("mechanism", ""))
    left_user = str(left.get("target_user", ""))
    right_desc = str(right.get("description", ""))
    right_mech = str(right.get("mechanism", ""))
    right_user = str(right.get("target_user", ""))
    
    prompt = f"""
Idea A: {left_desc} / {left_mech} (Focus: {left_user})
Idea B: {right_desc} / {right_mech} (Focus: {right_user})

Identify the most fundamental 'strategic contradiction' or 'orthogonal tension' between these two ideas in a single short sentence.
Output the conflict in {language}.
"""
    try:
        payload = client.chat_json(
            user_prompt=prompt,
            system_prompt="Return exactly one JSON object with a single key 'conflict' containing the 1-sentence contradiction. Return only valid JSON.",
            debug_label="combiner_conflict_extractor",
            num_predict=100,
        )
        conflict = str(payload.get("conflict", "")).strip()
        if conflict:
            return conflict
        return "Unknown conflict"
    except Exception as exc:
        print(f"[combiner] Failed to extract dynamic conflict: {exc}")
        return "Unknown conflict"


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
