"""Idea mutation for expanding the search space."""

from __future__ import annotations

import json
import time
from typing import Any

from config import (
    DEFAULT_OUTPUT_LANGUAGE,
    MUTATION_COUNT,
    MUTATOR_ENFORCE_FAMILY,
    MUTATION_PROMPT,
    MUTATOR_NUM_PREDICT,
    OLLAMA_MODEL,
)
from core.utils import load_text
from llm.ollama_client import OllamaClient


MUTATION_TYPES = [
    "constraint_removal",
    "reverse_assumption",
    "cross_domain",
    "simplification",
    "exaggeration",
    "persona_shift",
]

MUTATION_FAMILIES = {
    "constraint_removal": "remove a major constraint, friction, or rule",
    "reverse_assumption": "invert a core assumption or default logic",
    "cross_domain": "import a mechanism from a different domain",
    "simplification": "reduce the idea to a smaller, leaner operating model",
    "exaggeration": "push one mechanism to an extreme version",
    "persona_shift": "reinterpret the idea from a sharply different persona worldview",
}


def build_mutation_prompt(
    problem: str,
    idea: dict,
    mutation_count: int,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
    existing_mutations: list[dict] | None = None,
    required_family: str | None = None,
) -> str:
    """Build a strict prompt for strategy-level mutation."""
    base_prompt = load_text(MUTATION_PROMPT)
    parent_depth = int(idea.get("depth") or 0)
    parent_persona = str(idea.get("persona") or idea.get("source_persona") or "").strip()
    existing_mutations = existing_mutations or []

    rendered = base_prompt.format(
        id=idea.get("id", "idea_0"),
        title=idea.get("title", ""),
        strategy_type=idea.get("strategy_type", "general"),
        description=idea.get("description", ""),
        language=language,
        mutation_count=mutation_count,
    )

    instructions = f"""

Original problem:
{problem}

Parent idea metadata:
- persona: {parent_persona or "unknown"}
- parent_id: {idea.get("id", "idea_0")}
- parent_depth: {parent_depth}

Mutation rules:
- Generate exactly {mutation_count} mutated ideas.
- Each mutation must be strategically different from the parent idea.
- Do not paraphrase or lightly rename the same concept.
- Change the underlying approach, mechanism, assumption, or operating model.
- Attack the problem from a noticeably different angle than the parent.
- Avoid the same product shape, interface pattern, business model, or interaction loop as the parent.
- Make the mutations different from each other as well.
- Novelty matters more than realism.
- Some mutations may be strange, speculative, or partially unrealistic, and that is acceptable.
- Do not soften an idea into a safer version if a more surprising direction is available.
- Keep each description concise but concrete.
- Prefer mutation types such as: {", ".join(MUTATION_TYPES)}.
- Write all natural-language fields in {language}.
- Return only a JSON object. Do not wrap it in markdown or code fences.
""".strip()
    if required_family and MUTATOR_ENFORCE_FAMILY:
        instructions += (
            "\n"
            f"- The mutation must specifically use the `{required_family}` family: "
            f"{MUTATION_FAMILIES.get(required_family, required_family)}."
        )
    elif required_family:
        instructions += (
            "\n"
            f"- You may use the `{required_family}` family as inspiration, "
            "but you may ignore it if a more surprising mutation appears."
        )

    instructions += """

The JSON object must contain these required fields:
- title
- strategy_type
- description
- mutation_type

Optional if useful:
- persona
- mechanism
- target_user
- execution_context
- expected_advantage
""".strip()
    if existing_mutations:
        prior_lines = ["Already accepted mutations to avoid repeating:"]
        for mutation in existing_mutations:
            prior_lines.append(
                "- "
                f"{mutation.get('mutation_type', 'unknown')} / "
                f"{mutation.get('strategy_type', 'general')} / "
                f"{mutation.get('title', 'untitled')}"
            )
        prior_block = "\n" + "\n".join(prior_lines)
    else:
        prior_block = ""
    return f"{rendered}\n\n{instructions}{prior_block}"


def mutate_idea(
    problem: str | dict,
    idea: dict | None = None,
    model: str = OLLAMA_MODEL,
    mutation_count: int = MUTATION_COUNT,
    language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> list[dict]:
    """Generate mutated ideas for one parent idea.

    Supports the new interface:
    - mutate_idea(problem, idea, model, mutation_count)

    Also tolerates the older project call style:
    - mutate_idea(idea)
    """
    if idea is None and isinstance(problem, dict):
        idea = problem
        problem = str(idea.get("description") or idea.get("title") or "")

    if not isinstance(problem, str) or not isinstance(idea, dict):
        raise RuntimeError("mutate_idea expects a problem string and an idea dict.")

    parent_id = str(idea.get("id") or "idea_0")
    parent_title = str(idea.get("title") or idea.get("strategy_type") or parent_id)
    print(
        f"[mutator] Mutating {parent_id} with model={model} "
        f"target_count={mutation_count} title={parent_title}"
    )
    started_at = time.perf_counter()
    valid_mutations: list[dict] = []
    max_attempts = max(4, mutation_count * 5)
    for attempt_index in range(1, max_attempts + 1):
        if len(valid_mutations) >= mutation_count:
            break

        print(
            f"[mutator] Attempt {attempt_index}/{max_attempts} for {parent_id} "
            f"(accepted={len(valid_mutations)}/{mutation_count})"
        )
        required_family = _choose_mutation_family(valid_mutations)
        prompt = build_mutation_prompt(
            problem,
            idea,
            1,
            language=language,
            existing_mutations=valid_mutations,
            required_family=required_family,
        )
        raw_output = _generate_text(model=model, prompt=prompt, language=language)
        parsed_candidates = _parse_mutation_payload(raw_output)
        print(
            f"[mutator] Parsed {len(parsed_candidates)} raw mutation candidates "
            f"for {parent_id} on attempt {attempt_index}."
        )
        mutations = normalize_mutation_output(
            raw_output,
            idea,
            start_index=len(valid_mutations) + 1,
        )
        print(
            f"[mutator] Normalized {len(mutations)} mutation candidates "
            f"for {parent_id} on attempt {attempt_index}."
        )

        for mutation in mutations:
            is_valid, reason = validate_mutation(mutation, idea, valid_mutations)
            if is_valid:
                valid_mutations.append(mutation)
                print(
                    f"[mutator] Accepted mutation {mutation['id']} "
                    f"({mutation.get('mutation_type', 'unknown')})"
                )
            else:
                print(
                    f"[mutator] Rejected mutation {mutation.get('id', 'unknown')} "
                    f"reason={reason}"
                )
            if len(valid_mutations) == mutation_count:
                break

    elapsed = time.perf_counter() - started_at
    print(
        f"[mutator] Completed {parent_id} in {elapsed:.1f}s "
        f"with {len(valid_mutations)} valid mutations."
    )
    return valid_mutations


def normalize_mutation_output(
    raw_output: Any,
    parent_idea: dict,
    *,
    start_index: int = 1,
) -> list[dict]:
    """Normalize model output into JSON-safe mutation objects."""
    parsed = _parse_mutation_payload(raw_output)
    if not isinstance(parsed, list):
        print("[mutator] Parsed payload is not a list. Returning empty normalized output.")
        return []

    parent_id = str(parent_idea.get("id") or "idea_0")
    parent_persona = str(
        parent_idea.get("persona") or parent_idea.get("source_persona") or ""
    ).strip()
    depth = int(parent_idea.get("depth") or 0) + 1

    normalized: list[dict] = []
    for index, item in enumerate(parsed, start=start_index):
        if not isinstance(item, dict):
            continue

        mutation = {
            "id": str(item.get("id") or f"{parent_id}_mut_{index}"),
            "persona": str(item.get("persona") or parent_persona).strip(),
            "title": str(item.get("title") or f"Mutation {index}").strip(),
            "strategy_type": str(item.get("strategy_type") or "general").strip(),
            "description": str(item.get("description") or "").strip(),
            "mechanism": " ".join(str(item.get("mechanism") or "").split()),
            "target_user": " ".join(str(item.get("target_user") or "").split()),
            "execution_context": " ".join(
                str(item.get("execution_context") or "").split()
            ),
            "expected_advantage": " ".join(
                str(item.get("expected_advantage") or "").split()
            ),
            "mutation_type": str(item.get("mutation_type") or "variation").strip(),
            "parent_id": parent_id,
            "depth": depth,
        }

        if mutation["description"] and mutation["title"]:
            normalized.append(mutation)
        else:
            print(
                f"[mutator] Dropped candidate {mutation['id']} during normalization "
                f"because title or description was empty."
            )

    return normalized


def validate_mutation(
    idea: dict,
    parent_idea: dict,
    sibling_mutations: list[dict] | None = None,
) -> tuple[bool, str]:
    """Reject obvious duplicates and weak mutations."""
    description = str(idea.get("description") or "").strip().lower()
    title = str(idea.get("title") or "").strip().lower()
    strategy_type = str(idea.get("strategy_type") or "").strip().lower()
    mutation_type = str(idea.get("mutation_type") or "").strip().lower()
    sibling_mutations = sibling_mutations or []

    parent_description = str(parent_idea.get("description") or "").strip().lower()
    parent_title = str(parent_idea.get("title") or "").strip().lower()
    parent_strategy = str(parent_idea.get("strategy_type") or "").strip().lower()

    if not description or not title:
        return False, "missing_title_or_description"
    if title == parent_title and description == parent_description:
        return False, "same_title_and_description_as_parent"
    if description == parent_description:
        return False, "same_description_as_parent"
    overlap = _token_overlap(description, parent_description)
    if strategy_type == parent_strategy and overlap > 0.72:
        return False, "same_strategy_and_high_overlap"
    if overlap > 0.82:
        return False, "too_close_to_parent"
    if mutation_type in {"", "variation", "same"} and _token_overlap(description, parent_description) > 0.7:
        return False, "weak_mutation_type_and_high_overlap"
    if (
        MUTATOR_ENFORCE_FAMILY
        and mutation_type in {
            str(item.get("mutation_type") or "").strip().lower()
            for item in sibling_mutations
        }
    ):
        return False, "duplicate_mutation_family"
    for sibling in sibling_mutations:
        sibling_title = str(sibling.get("title") or "").strip().lower()
        sibling_description = str(sibling.get("description") or "").strip().lower()
        sibling_strategy = str(sibling.get("strategy_type") or "").strip().lower()
        if title == sibling_title:
            return False, "same_title_as_existing_mutation"
        if strategy_type == sibling_strategy and _token_overlap(description, sibling_description) > 0.82:
            return False, "same_strategy_as_existing_mutation"
    return True, "accepted"


def _choose_mutation_family(existing_mutations: list[dict]) -> str:
    used = {
        str(item.get("mutation_type") or "").strip().lower()
        for item in existing_mutations
    }
    available = [family for family in MUTATION_TYPES if family not in used]
    if available:
        return available[0]
    return MUTATION_TYPES[len(existing_mutations) % len(MUTATION_TYPES)]


def _generate_text(model: str, prompt: str, language: str) -> str:
    """Use the existing Ollama client and return raw text content."""
    client = OllamaClient(model=model)
    body = client._post(
        {
            "model": model,
            "stream": True,
            "options": {"num_predict": MUTATOR_NUM_PREDICT},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strategic idea mutator. "
                        "Novelty matters more than realism. "
                        "Speculative or partially unrealistic ideas are acceptable if they open a new direction. "
                        "A good mutation should feel like a different attack on the problem, not a polished version of the parent. "
                        "Avoid repeating the same product form, interaction pattern, or business model as the parent. "
                        "Keep each mutation concise. "
                        f"Write all natural-language fields in {language}. "
                        "Return only a single JSON object. "
                        "Do not use markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        stream=True,
        debug_label=f"mutator model={model}",
    )
    return client._extract_message_content(body)


def _parse_mutation_payload(raw_output: Any) -> list[dict]:
    """Parse mutations from native objects or loosely formatted text."""
    if isinstance(raw_output, list):
        return [item for item in raw_output if isinstance(item, dict)]

    if isinstance(raw_output, dict):
        for key in ("mutations", "ideas", "items"):
            value = raw_output.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [raw_output]

    if not isinstance(raw_output, str):
        return []

    text = _strip_code_fences(raw_output.strip())
    if not text:
        print("[mutator] Raw output was empty.")
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("[mutator] Direct JSON parse failed. Trying first-array extraction.")
        parsed = _extract_first_json_array(text)

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return _parse_mutation_payload(parsed)
    print("[mutator] Parsed output was neither list nor dict.")
    return []


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences when the model ignores instructions."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_array(text: str) -> Any:
    """Extract the first JSON array from a noisy model response."""
    start = text.find("[")
    if start == -1:
        print("[mutator] Could not find JSON array start '[' in raw output.")
        return []

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    print("[mutator] Could not recover a complete JSON array from raw output.")
    return []


def _token_overlap(left: str, right: str) -> float:
    """Very small lexical overlap heuristic for duplicate rejection."""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
