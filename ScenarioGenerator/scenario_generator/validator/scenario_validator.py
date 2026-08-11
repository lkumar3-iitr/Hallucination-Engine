from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scenario_generator.schema import ResolvedScenario, ScenarioSpec


@dataclass
class ValidationIssue:
    severity: str  # "warning" or "error"
    message: str


@dataclass
class ValidationResult:
    ok: bool
    issues: List[ValidationIssue]


class ScenarioValidator:
    """Basic scenario validator v1."""

    def validate_structured(self, scenario: ScenarioSpec) -> ValidationResult:
        issues: list[ValidationIssue] = []

        actor_ids = [actor.actor_id for actor in scenario.actors]
        if len(actor_ids) != len(set(actor_ids)):
            issues.append(ValidationIssue("error", "actor_id values must be unique."))

        for actor in scenario.actors:
            if actor.initial_x_m < -10:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"{actor.actor_id} starts far behind ego: x={actor.initial_x_m:.2f} m",
                    )
                )

        return ValidationResult(
            ok=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )

    def validate_resolved(self, scenario: ResolvedScenario) -> ValidationResult:
        issues: list[ValidationIssue] = []

        for frame in scenario.frames:
            if frame.x_m < -20:
                issues.append(
                    ValidationIssue(
                        "warning",
                        f"{frame.actor_id} moves far behind ego at frame {frame.frame_idx}.",
                    )
                )

        return ValidationResult(
            ok=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )
