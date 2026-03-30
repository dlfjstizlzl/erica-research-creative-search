"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Idea:
    """One generated or mutated idea."""

    id: str
    strategy_type: str
    description: str
    title: str = ""
    persona: str = ""
    mechanism: str = ""
    target_user: str = ""
    execution_context: str = ""
    expected_advantage: str = ""
    parent_id: str = ""
    depth: int = 0
    mutation_type: str = ""
    source_model: str = ""
    source_persona: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    score_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Idea":
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            persona=str(data.get("persona") or ""),
            strategy_type=str(data.get("strategy_type") or "general"),
            description=str(data.get("description") or ""),
            mechanism=str(data.get("mechanism") or ""),
            target_user=str(data.get("target_user") or ""),
            execution_context=str(data.get("execution_context") or ""),
            expected_advantage=str(data.get("expected_advantage") or ""),
            parent_id=str(data.get("parent_id") or ""),
            depth=int(data.get("depth") or 0),
            mutation_type=str(data.get("mutation_type") or ""),
            source_model=str(data.get("source_model") or ""),
            source_persona=str(data.get("source_persona") or ""),
            scores={
                str(key): float(value)
                for key, value in dict(data.get("scores") or {}).items()
                if isinstance(value, (int, float))
            },
            score_meta=dict(data.get("score_meta") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "strategy_type": self.strategy_type,
            "description": self.description,
        }
        if self.title:
            payload["title"] = self.title
        if self.persona:
            payload["persona"] = self.persona
        if self.mechanism:
            payload["mechanism"] = self.mechanism
        if self.target_user:
            payload["target_user"] = self.target_user
        if self.execution_context:
            payload["execution_context"] = self.execution_context
        if self.expected_advantage:
            payload["expected_advantage"] = self.expected_advantage
        if self.parent_id:
            payload["parent_id"] = self.parent_id
        if self.depth:
            payload["depth"] = self.depth
        if self.mutation_type:
            payload["mutation_type"] = self.mutation_type
        if self.source_model:
            payload["source_model"] = self.source_model
        if self.source_persona:
            payload["source_persona"] = self.source_persona
        if self.scores:
            payload["scores"] = self.scores
        if self.score_meta:
            payload["score_meta"] = self.score_meta
        return payload


@dataclass
class PipelineResult:
    """Serializable pipeline output."""

    problem: str
    output_language: str = ""
    base_ideas: list[Idea] = field(default_factory=list)
    filtered_ideas: list[Idea] = field(default_factory=list)
    mutated_ideas: list[Idea] = field(default_factory=list)
    best_idea: Idea | None = None
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "problem": self.problem,
            "output_language": self.output_language,
            "base_ideas": [idea.to_dict() for idea in self.base_ideas],
            "filtered_ideas": [idea.to_dict() for idea in self.filtered_ideas],
            "mutated_ideas": [idea.to_dict() for idea in self.mutated_ideas],
        }
        if self.best_idea is not None:
            payload["best_idea"] = self.best_idea.to_dict()
        return payload
