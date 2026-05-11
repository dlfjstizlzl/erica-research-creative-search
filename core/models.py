"""Shared data models."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def generate_short_id() -> str:
    return uuid.uuid4().hex[:8]


class Idea(BaseModel):
    """One generated or mutated idea."""

    id: str = Field(default_factory=generate_short_id)
    strategy_type: str = "general"
    description: str = ""
    title: str = ""
    persona: str = ""
    mechanism: str = ""
    target_user: str = ""
    execution_context: str = ""
    expected_advantage: str = ""
    parent_id: str = ""
    parent_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    mutation_type: str = ""
    combination_type: str = ""
    origin_type: str = ""
    generation: int = 0
    source_model: str = ""
    source_persona: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    score_meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Idea":
        """Backwards compatibility wrapper. Pydantic handles validation."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding defaults to keep JSON small if desired, 
        but we'll just dump everything for consistent schema."""
        return self.model_dump()


class PipelineResult(BaseModel):
    """Serializable pipeline output."""

    problem: str
    reframed_problem: str = ""
    output_language: str = ""
    base_ideas: list[Idea] = Field(default_factory=list)
    combined_ideas: list[Idea] = Field(default_factory=list)
    filtered_ideas: list[Idea] = Field(default_factory=list)
    mutated_ideas: list[Idea] = Field(default_factory=list)
    best_practical: Optional[Idea] = None
    best_balanced: Optional[Idea] = None
    best_wild: Optional[Idea] = None
    archive: list[dict[str, Any]] = Field(default_factory=list)
    archive_summary: dict[str, Any] = Field(default_factory=dict)
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
