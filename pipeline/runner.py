"""Pipeline orchestration."""

from __future__ import annotations

from config import (
    COMBINATION_PAIR_COUNT,
    COMBINER_MODEL,
    DEFAULT_OUTPUT_LANGUAGE,
    OLLAMA_MODEL,
    MUTATOR_MODEL,
    PARENT_SELECTION_COUNT,
    POOL_MAX_SIZE,
    PROBLEMS_FILE,
    REFRAMER_MODEL,
    RESULTS_DIR,
    SEARCH_MAX_GENERATIONS,
)
from core.models import Idea, PipelineResult
from core.utils import load_json, save_json, timestamp_slug
from pipeline.archive import (
    initialize_archive,
    mark_selection_in_archive,
    summarize_archive,
    update_archive,
)
from pipeline.combiner import combine_ideas
from pipeline.filter import filter_diverse_ideas
from pipeline.generator import generate_base_ideas
from pipeline.mutator import mutate_idea
from pipeline.pool import initialize_pool, update_pool
from pipeline.problem_reframer import reframe_problem
from pipeline.scoring import score_ideas
from pipeline.selection import (
    select_combination_pairs,
    select_final_bests,
    select_parent_ideas,
)


WORKING_LANGUAGE = "English"


def load_problem_from_file(index: int = 0) -> str:
    """Load one problem from data/problems.json."""
    payload = load_json(PROBLEMS_FILE)
    if isinstance(payload, dict):
        problems = payload.get("problems", [])
    else:
        problems = payload

    if not isinstance(problems, list) or not problems:
        raise RuntimeError("No problems found in data/problems.json.")
    if index < 0 or index >= len(problems):
        raise RuntimeError(f"Problem index {index} is out of range.")

    return str(problems[index])


def run_pipeline(problem: str, language: str = DEFAULT_OUTPUT_LANGUAGE) -> dict[str, object]:
    """Run seed generation and iterative search loop."""
    raw_problem = problem

    print("[runner] Step 1/7: Reframing problem input.")
    reframed_problem = reframe_problem(
        raw_problem,
        model=REFRAMER_MODEL,
        language=WORKING_LANGUAGE,
    )
    search_problem = _build_search_problem_context(raw_problem, reframed_problem)
    print(f"[runner] Raw problem: {raw_problem}")
    print(f"[runner] Reframed problem: {reframed_problem}")
    print("[runner] Using both raw and reframed problem as search context.")

    print("[runner] Step 2/7: Generating base ideas.")
    seed_dicts = [
        _seedify(item, index=index)
        for index, item in enumerate(
            generate_base_ideas(search_problem, language=WORKING_LANGUAGE),
            start=1,
        )
    ]
    seed_dicts = _ensure_unique_ids(seed_dicts)
    print(f"[runner] Generated {len(seed_dicts)} seed ideas.")

    print("[runner] Step 3/7: Initial scoring and pool setup.")
    scored_seed_dicts = score_ideas(search_problem, seed_dicts)
    base_ideas = [Idea.from_dict(item) for item in scored_seed_dicts]
    active_pool = initialize_pool(scored_seed_dicts, max_size=POOL_MAX_SIZE)
    archive = initialize_archive(scored_seed_dicts)
    mutated_ideas: list[Idea] = []
    combined_ideas: list[Idea] = []
    print(f"[runner] Initialized active pool with {len(active_pool)} ideas.")

    print("[runner] Step 4/7: Iterative search loop.")
    for generation in range(1, SEARCH_MAX_GENERATIONS + 1):
        print(f"[runner] Generation {generation}/{SEARCH_MAX_GENERATIONS}")
        parent_dicts = select_parent_ideas(
            active_pool,
            exploit_count=max(1, PARENT_SELECTION_COUNT - 1),
            explore_count=1,
        )
        print(f"[runner] Selected {len(parent_dicts)} mutation parents.")

        new_mutation_dicts: list[dict] = []
        for parent in parent_dicts:
            parent_id = str(parent.get("id") or "")
            print(f"[runner] Mutating parent: {parent_id}")
            mutation_outputs = mutate_idea(
                search_problem,
                parent,
                model=MUTATOR_MODEL,
                language=WORKING_LANGUAGE,
            )
            for item in mutation_outputs:
                new_mutation_dicts.append(_mutationify(item, generation=generation))

        parent_pairs = select_combination_pairs(
            active_pool,
            max_pairs=COMBINATION_PAIR_COUNT,
        )
        print(f"[runner] Selected {len(parent_pairs)} recombination pairs.")
        new_combination_dicts = combine_ideas(
            search_problem,
            parent_pairs,
            model=COMBINER_MODEL,
            language=WORKING_LANGUAGE,
            generation=generation,
        )

        new_candidate_dicts = [*new_mutation_dicts, *new_combination_dicts]
        if not new_candidate_dicts:
            print("[runner] No new candidates produced in this generation.")
            continue

        print(
            f"[runner] Generation {generation} produced "
            f"{len(new_mutation_dicts)} mutations and {len(new_combination_dicts)} combinations."
        )
        print(f"[runner] Scoring {len(new_candidate_dicts)} new candidates.")
        scoring_context = [*active_pool, *new_candidate_dicts]
        scored_context = score_ideas(search_problem, scoring_context)
        new_candidate_ids = {
            str(item.get("id") or "")
            for item in new_candidate_dicts
        }
        scored_new_candidates = [
            item
            for item in scored_context
            if str(item.get("id") or "") in new_candidate_ids
        ]
        active_pool = update_pool(
            active_pool,
            scored_new_candidates,
            max_size=POOL_MAX_SIZE,
        )
        archive = update_archive(
            archive,
            scored_new_candidates,
            active_ids={str(item.get('id') or '') for item in active_pool},
        )

        mutated_ideas.extend(
            Idea.from_dict(item)
            for item in scored_new_candidates
            if str(item.get("origin_type") or "") == "mutation"
        )
        combined_ideas.extend(
            Idea.from_dict(item)
            for item in scored_new_candidates
            if str(item.get("origin_type") or "") == "combination"
        )
        print(f"[runner] Active pool size after generation {generation}: {len(active_pool)}")

    print("[runner] Step 5/7: Final pool filtering.")
    filtered_dicts = filter_diverse_ideas(active_pool)
    filtered_ideas = [Idea.from_dict(item) for item in filtered_dicts]
    print(f"[runner] Final filtered pool size: {len(filtered_ideas)}")

    print("[runner] Step 6/7: Final ranking.")
    final_bests = select_final_bests(filtered_dicts)
    best_practical = _idea_from_optional_dict(final_bests.get("best_practical"))
    best_balanced = _idea_from_optional_dict(final_bests.get("best_balanced"))
    best_wild = _idea_from_optional_dict(final_bests.get("best_wild"))
    archive = mark_selection_in_archive(
        archive,
        best_practical_id=best_practical.id if best_practical else "",
        best_balanced_id=best_balanced.id if best_balanced else "",
        best_wild_id=best_wild.id if best_wild else "",
        active_ids={idea.id for idea in filtered_ideas},
    )
    archive_summary = summarize_archive(archive)
    if best_practical is not None:
        print(f"[runner] Best practical: {best_practical.id} ({best_practical.title})")
    if best_balanced is not None:
        print(f"[runner] Best balanced: {best_balanced.id} ({best_balanced.title})")
    if best_wild is not None:
        print(f"[runner] Best wild: {best_wild.id} ({best_wild.title})")

    print("[runner] Step 7/7: Saving final result to JSON.")
    output_path = RESULTS_DIR / f"run_{timestamp_slug()}.json"
    result = PipelineResult(
        problem=raw_problem,
        reframed_problem=reframed_problem,
        output_language=WORKING_LANGUAGE,
        base_ideas=base_ideas,
        combined_ideas=combined_ideas,
        filtered_ideas=filtered_ideas,
        mutated_ideas=mutated_ideas,
        best_practical=best_practical,
        best_balanced=best_balanced,
        best_wild=best_wild,
        archive=archive,
        archive_summary=archive_summary,
        output_path=str(output_path),
    )
    save_json(output_path, result.to_dict())
    print(f"[runner] Result saved to: {output_path}")
    return result.to_dict()


def _seedify(idea: dict, *, index: int) -> dict:
    normalized = dict(idea)
    normalized["origin_type"] = "base"
    normalized["generation"] = 0
    normalized["depth"] = int(normalized.get("depth") or 0)
    normalized["parent_ids"] = []
    normalized["id"] = str(normalized.get("id") or f"seed_{index}").strip() or f"seed_{index}"
    return normalized


def _mutationify(idea: dict, *, generation: int) -> dict:
    normalized = dict(idea)
    normalized["origin_type"] = "mutation"
    normalized["generation"] = generation
    parent_id = str(normalized.get("parent_id") or "").strip()
    normalized["parent_ids"] = [parent_id] if parent_id else []
    mutation_id = str(normalized.get("id") or "").strip()
    if mutation_id:
        normalized["id"] = f"{mutation_id}_g{generation}"
    return normalized


def _ensure_unique_ids(ideas: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    normalized_items: list[dict] = []
    for index, idea in enumerate(ideas, start=1):
        item = dict(idea)
        base_id = str(item.get("id") or f"idea_{index}").strip() or f"idea_{index}"
        seen[base_id] = seen.get(base_id, 0) + 1
        if seen[base_id] > 1:
            item["id"] = f"{base_id}_{seen[base_id]}"
            print(
                f"[runner] Adjusted duplicate id {base_id} -> {item['id']}"
            )
        else:
            item["id"] = base_id
        normalized_items.append(item)
    return normalized_items


def _idea_from_optional_dict(data: dict | None) -> Idea | None:
    if not isinstance(data, dict):
        return None
    return Idea.from_dict(data)


def _build_search_problem_context(raw_problem: str, reframed_problem: str) -> str:
    raw_text = " ".join(str(raw_problem or "").split()).strip()
    reframed_text = " ".join(str(reframed_problem or "").split()).strip()
    if not reframed_text or reframed_text == raw_text:
        return raw_text
    return (
        f"Original problem:\n{raw_text}\n\n"
        f"Reframed problem:\n{reframed_text}"
    )
