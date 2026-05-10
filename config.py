"""Project configuration."""

from __future__ import annotations

import os
from os import cpu_count
from pathlib import Path
from typing import Final


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    """Parse a simple comma-separated env var."""
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_bool_env(name: str, default: bool) -> bool:
    """Parse a permissive boolean env var."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


load_dotenv(ENV_FILE)

DEFAULT_GENERATOR_PERSONAS: Final[list[str]] = [
    "systems architect: obsessed with elegant systems, modular infrastructure, and designs that still work at 100x scale",
    "contrarian founder: rejects conventional solutions, hunts asymmetric advantages, and prefers bold bets over safe optimization",
    "behavioral scientist: sees every problem as a battle of habits, incentives, cognitive bias, and repeated decision loops",
    "operations commander: prioritizes ruthless clarity, standard operating procedures, throughput, and failure-resistant execution",
    "community cult-builder: designs belonging, ritual, identity, and strong network effects without crossing ethical lines",
    "product auteur: cares intensely about emotional experience, interaction detail, and products people become attached to",
    "growth hacker: optimizes for compounding loops, virality, activation triggers, and low-cost distribution leverage",
    "master teacher: turns any solution into a structured journey of scaffolding, mastery, and reinforcing feedback",
    "anthropologist on the ground: studies hidden rituals, local norms, symbols, and lived behavior before intervening",
    "game designer: reframes boring systems as progression engines full of challenge, reward, suspense, and momentum",
    "policy operator: thinks through incentives, governance, compliance, public legitimacy, and second-order effects",
    "zero-budget survival engineer: assumes almost no money, no trust, limited tools, and still must make it work",
    "luxury concierge strategist: designs white-glove, high-status, high-delight experiences that feel exclusive and premium",
    "sales machine operator: focuses on persuasion systems, conversion flows, repeatable scripts, and objection handling",
    "data scientist: trusts measurement over opinion, models uncertainty, and continuously adapts based on evidence",
    "climate resilience planner: optimizes for sustainability, durability, energy efficiency, and long-term resilience",
    "provocative artist: values surprise, symbolic meaning, emotional charge, and culturally memorable form factors",
    "security pessimist: assumes failure, abuse, misuse, adversaries, edge cases, and fragile trust by default",
    "grassroots organizer: builds through trust, collective action, mutual aid, and participation from the bottom up",
    "hardcore futurist: thinks in 10-year shifts, emerging norms, nonlinear change, and strange but plausible futures",
    "municipal sanitation operator: thinks in collection routes, contamination rates, shift constraints, maintenance realities, and service uptime",
    "public procurement lead: optimizes around budgets, contracts, rollout risk, vendor reliability, and measurable public value",
    "factory process engineer: redesigns bottlenecks, throughput, instrumentation, and handoff points with minimal waste and variance",
    "service designer: maps backstage operations, frontstage touchpoints, user friction, and operational consistency",
    "public health inspector: cares about safety, compliance, hygiene behavior, risk communication, and intervention practicality",
    "warehouse logistics planner: focuses on sorting flows, staging, routing, load balancing, and error-proof movement of materials",
    "maintenance technician: prefers durable systems with simple failure modes, repairability, and low training overhead",
    "cooperative manager: designs shared incentives, member participation, pooled resources, and durable governance mechanisms",
    "school administrator: balances behavior change, limited staff attention, incentives, and repeatable routines in messy real settings",
    "retail operator: sees problems through customer behavior, inventory movement, shrinkage, incentives, and frontline execution",
]

BASE_GENERATION_PROMPT = PROMPTS_DIR / "base_generation.txt"
PROBLEM_REFRAMING_PROMPT = PROMPTS_DIR / "problem_reframing.txt"
MUTATION_PROMPT = PROMPTS_DIR / "mutation.txt"
COMBINATION_PROMPT = PROMPTS_DIR / "combination.txt"
DIVERSITY_FILTER_PROMPT = PROMPTS_DIR / "diversity_filter.txt"
PROBLEMS_FILE = DATA_DIR / "problems.json"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_GENERATOR_MODELS = parse_csv_env("OLLAMA_GENERATOR_MODELS", [OLLAMA_MODEL])
COMBINER_MODEL = os.getenv("COMBINER_MODEL", OLLAMA_MODEL)
MUTATOR_MODEL = os.getenv("MUTATOR_MODEL", OLLAMA_MODEL)
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", OLLAMA_MODEL)
REFRAMER_MODEL = os.getenv("REFRAMER_MODEL", COMBINER_MODEL)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
DEFAULT_OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "Korean")
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
REFRAMER_NUM_PREDICT = int(os.getenv("REFRAMER_NUM_PREDICT", "220"))
GENERATOR_NUM_PREDICT = int(os.getenv("GENERATOR_NUM_PREDICT", "220"))
MUTATOR_NUM_PREDICT = int(os.getenv("MUTATOR_NUM_PREDICT", "260"))

GENERATOR_PERSONAS = parse_csv_env("GENERATOR_PERSONAS", DEFAULT_GENERATOR_PERSONAS)
GENERATOR_SAMPLE_COUNT = int(os.getenv("GENERATOR_SAMPLE_COUNT", "5"))
GENERATOR_MAX_WORKERS = int(
    os.getenv("GENERATOR_MAX_WORKERS", str(min(3, max(1, cpu_count() or 1))))
)
MUTATOR_ENFORCE_FAMILY = parse_bool_env("MUTATOR_ENFORCE_FAMILY", False)
FILTER_USE_AGGRESSIVE_PRUNE = parse_bool_env("FILTER_USE_AGGRESSIVE_PRUNE", False)
FILTER_SIMILARITY_THRESHOLD = float(os.getenv("FILTER_SIMILARITY_THRESHOLD", "0.82"))
SCORING_NEIGHBOR_COUNT = int(os.getenv("SCORING_NEIGHBOR_COUNT", "3"))

BASE_IDEA_COUNT = 5
MUTATION_COUNT = int(os.getenv("MUTATION_COUNT", "1"))
FILTER_KEEP_COUNT = 3
SEARCH_MAX_GENERATIONS = int(os.getenv("SEARCH_MAX_GENERATIONS", "2"))
PARENT_SELECTION_COUNT = int(os.getenv("PARENT_SELECTION_COUNT", "3"))
COMBINATION_PAIR_COUNT = int(os.getenv("COMBINATION_PAIR_COUNT", "2"))
POOL_MAX_SIZE = int(os.getenv("POOL_MAX_SIZE", "10"))
POOL_MAX_PER_STRATEGY = int(os.getenv("POOL_MAX_PER_STRATEGY", "2"))
