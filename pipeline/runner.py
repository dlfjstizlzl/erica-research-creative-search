"""Orchestrator for the evolutionary search pipeline."""

from __future__ import annotations

import asyncio
import json
import time

from config import (
    DEFAULT_OUTPUT_LANGUAGE,
    PARENT_SELECTION_COUNT,
    POOL_MAX_SIZE,
    PROBLEMS_FILE,
    SEARCH_MAX_GENERATIONS,
)
from core.models import Idea, PipelineResult
from pipeline.combiner import combine_ideas
from pipeline.generator import generate_base_ideas
from pipeline.mutator import mutate_idea
from pipeline.problem_reframer import reframe_problem
from pipeline.scoring import score_ideas
from pipeline.selection import (
    select_combination_pairs,
    select_final_bests,
    select_parent_ideas,
)

WORKING_LANGUAGE = DEFAULT_OUTPUT_LANGUAGE


def load_problem_from_file(index: int = 0) -> str:
    """Load a sample problem from data/problems.json."""
    if not PROBLEMS_FILE.exists():
        raise FileNotFoundError(f"Missing {PROBLEMS_FILE}")
    data = json.loads(PROBLEMS_FILE.read_text(encoding="utf-8"))
    problems = data.get("problems", [])
    if not problems:
        raise ValueError(f"No problems found in {PROBLEMS_FILE}")
    idx = index % len(problems)
    print(f"[runner] Loaded problem {idx} from data/problems.json")
    return str(problems[idx])


async def run_pipeline(search_problem: str) -> dict:
    """Execute the async creative search pipeline."""
    print("\n" + "=" * 60)
    print("🚀  CREATIVE SEARCH PIPELINE START")
    print("=" * 60)
    print(f"[runner] Input problem: {search_problem}")
    started_at = time.perf_counter()

    reframe_start = time.perf_counter()
    reframed = await reframe_problem(search_problem, language=WORKING_LANGUAGE)
    print(f"[runner] Reframed in {time.perf_counter() - reframe_start:.1f}s")

    print("\n" + "-" * 60)
    print("🌱  STAGE 1: Base Generation")
    gen_start = time.perf_counter()
    base_ideas = await generate_base_ideas(reframed, language=WORKING_LANGUAGE)
    if not base_ideas:
        print("[runner] Generation failed.")
        return {}

    base_ideas = await score_ideas(search_problem, base_ideas)
    print(f"[runner] Stage 1 (Base Gen) completed in {time.perf_counter() - gen_start:.1f}s")

    active_pool = list(base_ideas)
    archive = list(base_ideas)

    all_mutated = []
    all_combined = []

    evo_start = time.perf_counter()
    for generation in range(1, SEARCH_MAX_GENERATIONS + 1):
        gen_round_start = time.perf_counter()
        print("\n" + "-" * 60)
        print(f"🧬  STAGE 2: Evolution (Generation {generation}/{SEARCH_MAX_GENERATIONS})")

        parent_dicts = select_parent_ideas(active_pool, selection_count=PARENT_SELECTION_COUNT)
        generation_pool = []

        print(f"\n[runner] --- Gen {generation} Mutation ---")
        mutation_tasks = [
            mutate_idea(
                search_problem,
                parent,
                model=None,
                language=WORKING_LANGUAGE,
            )
            for parent in parent_dicts
        ]
        mutation_results = await asyncio.gather(*mutation_tasks)
        for res in mutation_results:
            if res:
                for item in res:
                    item["generation"] = generation
                    generation_pool.append(item)
                    all_mutated.append(item)

        print(f"\n[runner] --- Gen {generation} Recombination ---")
        combine_pairs = select_combination_pairs(active_pool)
        if combine_pairs:
            combined_results = await combine_ideas(
                search_problem,
                combine_pairs,
                language=WORKING_LANGUAGE,
            )
            for item in combined_results:
                item["generation"] = generation
                generation_pool.append(item)
                all_combined.append(item)

        print(f"\n[runner] --- Gen {generation} Scoring ---")
        generation_pool = await score_ideas(search_problem, generation_pool)

        active_pool.extend(generation_pool)
        archive.extend(generation_pool)

        print(f"\n[runner] --- Gen {generation} Selection (Niching) ---")
        active_pool = select_parent_ideas(active_pool, selection_count=POOL_MAX_SIZE)

        print(f"[runner] End of Gen {generation} ({time.perf_counter() - gen_round_start:.1f}s). Active pool size: {len(active_pool)}")

    print(f"\n[runner] Stage 2 (Evolution) completed in {time.perf_counter() - evo_start:.1f}s")

    print("\n" + "-" * 60)
    print("🏆  STAGE 3: Final Selection")
    
    final_bests = select_final_bests(active_pool)

    elapsed = time.perf_counter() - started_at
    print("=" * 60)
    print(f"✅  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"    - Reframing: {gen_start - reframe_start:.1f}s")
    print(f"    - Base Gen: {evo_start - gen_start:.1f}s")
    print(f"    - Evolution: {time.perf_counter() - evo_start:.1f}s")
    print(f"    - Ideas: {len(archive)} total ({len(base_ideas)} base, {len(all_mutated)} mutated, {len(all_combined)} combined)")
    print("=" * 60)

    try:
        result_obj = PipelineResult(
            problem=search_problem,
            reframed_problem=reframed,
            output_language=WORKING_LANGUAGE,
            base_ideas=[Idea.model_validate(d) for d in base_ideas],
            combined_ideas=[Idea.model_validate(d) for d in all_combined],
            mutated_ideas=[Idea.model_validate(d) for d in all_mutated],
            best_practical=Idea.model_validate(final_bests["best_practical"]) if final_bests.get("best_practical") else None,
            best_balanced=Idea.model_validate(final_bests["best_balanced"]) if final_bests.get("best_balanced") else None,
            best_wild=Idea.model_validate(final_bests["best_wild"]) if final_bests.get("best_wild") else None,
            archive=[Idea.model_validate(d).model_dump() for d in archive],
            archive_summary={
                "total_generated": len(archive),
                "generations": SEARCH_MAX_GENERATIONS,
                "elapsed_seconds": elapsed,
            },
        )
        return result_obj.to_dict()
    except Exception as exc:
        print(f"[runner] Failed to build PipelineResult via Pydantic: {exc}")
        return {"error": str(exc), "status": "failed_validation"}
