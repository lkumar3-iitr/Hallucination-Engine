from __future__ import annotations

from typing import Any

from scenario_generator.schema import ResolvedScenario


def resolved_to_carla_plan(scenario: ResolvedScenario) -> dict[str, Any]:
    """Placeholder for CARLA execution backend.

    Later this should map local BEV states to CARLA world transforms,
    spawn actors, and run synchronous stepping.
    """

    return {
        "scenario_id": scenario.scenario_id,
        "note": "CARLA backend placeholder. Implement world transform mapping later.",
    }
