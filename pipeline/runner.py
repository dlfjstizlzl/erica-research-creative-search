"""Pipeline orchestration."""

from __future__ import annotations

from config import DEFAULT_OUTPUT_LANGUAGE, OLLAMA_MODEL, PROBLEMS_FILE, RESULTS_DIR
from core.models import Idea, PipelineResult
from core.utils import load_json, save_json, timestamp_slug
from pipeline.filter import filter_diverse_ideas
from pipeline.generator import generate_base_ideas
from pipeline.mutator import mutate_idea
from pipeline.scoring import score_ideas


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
    """Run generation, mutation, selection, post-hoc scoring, and persistence."""
    print("[runner] Step 1/5: Generating base ideas.")
    base_ideas = [
        Idea.from_dict(item) for item in generate_base_ideas(problem, language=WORKING_LANGUAGE)
    ]
    print(f"[runner] Generated {len(base_ideas)} base ideas.")

    print("[runner] Step 2/5: Mutating each base idea.")
    mutated_ideas: list[Idea] = []
    for idea in base_ideas:
        print(f"[runner] Mutating base idea: {idea.id} ({idea.title or idea.strategy_type})")
        before_count = len(mutated_ideas)
        mutated_ideas.extend(
            Idea.from_dict(item)
            for item in mutate_idea(
                problem,
                idea.to_dict(),
                model=idea.source_model or OLLAMA_MODEL,
                language=WORKING_LANGUAGE,
            )
        )
        created = len(mutated_ideas) - before_count
        print(f"[runner] Created {created} mutations from {idea.id}.")
    print(f"[runner] Total mutated ideas: {len(mutated_ideas)}")

    candidate_ideas = base_ideas + mutated_ideas
    print("[runner] Step 3/5: Merging idea pool and lightly filtering duplicates.")
    print(f"[runner] Candidate idea pool size: {len(candidate_ideas)}")
    filtered_ideas = [
        Idea.from_dict(item)
        for item in filter_diverse_ideas([idea.to_dict() for idea in candidate_ideas])
    ]
    print(f"[runner] Filtered idea count: {len(filtered_ideas)}")

    print("[runner] Step 4/5: Scoring outputs and selecting best idea.")
    scored_candidate_dicts = score_ideas(problem, [idea.to_dict() for idea in candidate_ideas])
    scored_by_id = {str(item.get("id") or ""): item for item in scored_candidate_dicts}
    base_ideas = [
        Idea.from_dict(scored_by_id.get(idea.id, idea.to_dict()))
        for idea in base_ideas
    ]
    mutated_ideas = [
        Idea.from_dict(scored_by_id.get(idea.id, idea.to_dict()))
        for idea in mutated_ideas
    ]
    filtered_ideas = [
        Idea.from_dict(scored_by_id.get(idea.id, idea.to_dict()))
        for idea in filtered_ideas
    ]
    best_idea = _select_best_idea(filtered_ideas)
    print(f"[runner] Post-hoc scored {len(scored_candidate_dicts)} ideas.")
    if best_idea is not None:
        print(f"[runner] Best idea selected: {best_idea.id} ({best_idea.title})")

    print("[runner] Step 5/5: Saving final result to JSON.")
    output_path = RESULTS_DIR / f"run_{timestamp_slug()}.json"
    result = PipelineResult(
        problem=problem,
        output_language=WORKING_LANGUAGE,
        base_ideas=base_ideas,
        filtered_ideas=filtered_ideas,
        mutated_ideas=mutated_ideas,
        best_idea=best_idea,
        output_path=str(output_path),
    )
    save_json(output_path, result.to_dict())
    print(f"[runner] Result saved to: {output_path}")
    return result.to_dict()


def _select_best_idea(ideas: list[Idea]) -> Idea | None:
    if not ideas:
        return None
    return max(ideas, key=_best_idea_score)


def _best_idea_score(idea: Idea) -> tuple[float, float, float, int]:
    scores = idea.scores or {}
    creativity = float(scores.get("creativity", 0.0))
    novelty = float(scores.get("novelty", 0.0))
    problem_fit = float(scores.get("problem_fit", scores.get("relevance", 0.0)))
    description_len = len((idea.description or "").split())
    return creativity, novelty, problem_fit, description_len
