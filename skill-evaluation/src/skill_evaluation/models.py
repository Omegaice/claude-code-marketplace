"""Pydantic models for skill evaluation."""

from pydantic import BaseModel, Field


class EvaluationSpec(BaseModel):
    """Specification for evaluating a skill."""

    skills: list[str] = Field(
        description="List of skill names to load for this evaluation"
    )
    query: str = Field(
        description="The query/prompt to send to Claude with the skills loaded"
    )
    files: list[str] = Field(
        default_factory=list,
        description="List of file paths to make available during evaluation"
    )
    expected_behavior: list[str] = Field(
        description="List of expected behaviors that should be demonstrated"
    )


class EvaluationResult(BaseModel):
    """Result of a skill evaluation."""

    spec: EvaluationSpec
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="LLM judge's reasoning for the score")
