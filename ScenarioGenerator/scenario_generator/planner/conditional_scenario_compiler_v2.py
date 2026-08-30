from __future__ import annotations

from scenario_generator.planner.behavior_compiler_v2 import (
    SemanticCruiseUntilStepV2,
    SemanticAccelerateToStepV2,
    SemanticBrakeToStepV2,
    SemanticCruiseStepV2,
    SemanticHoldStepV2,
    SemanticHoldUntilStepV2,
    SemanticLaneChangeStepV2,
)
from scenario_generator.planner.event_conditions_v2 import (
    resolve_condition_time,
)
from scenario_generator.planner.semantic_scenario_v2 import (
    SemanticScenarioV2,
    compile_semantic_scenario,
)
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


EPS = 1e-9


def _conditional_step_start_times(
    behavior,
    resolved_hold_durations_s,
    resolved_cruise_durations_s,
):
    """
    Return local behavior start time for every conditional step.

    Times are relative to actor spawn_time_s.
    """

    current_t = 0.0
    current_speed = behavior.initial_speed_mps

    starts = {}

    for step_index, step in enumerate(
        behavior.steps
    ):

        if isinstance(
            step,
            SemanticCruiseStepV2,
        ):
            current_t += step.duration_s

            if step.speed_mps is not None:
                current_speed = step.speed_mps

            continue

        if isinstance(
            step,
            SemanticCruiseUntilStepV2,
        ):
            starts[step_index] = current_t

            if (
                step_index
                not in resolved_cruise_durations_s
            ):
                raise ValueError(
                    "Missing provisional duration for "
                    f"cruise_until step {step_index}."
                )

            current_t += (
                resolved_cruise_durations_s[
                    step_index
                ]
            )

            if step.speed_mps is not None:
                current_speed = step.speed_mps

            continue

        if isinstance(
            step,
            SemanticLaneChangeStepV2,
        ):
            current_t += step.duration_s
            continue

        if isinstance(
            step,
            SemanticHoldStepV2,
        ):
            current_t += step.duration_s
            current_speed = 0.0
            continue

        if isinstance(
            step,
            SemanticHoldUntilStepV2,
        ):
            starts[step_index] = current_t

            if (
                step_index
                not in resolved_hold_durations_s
            ):
                raise ValueError(
                    "Missing provisional duration for "
                    f"hold_until step {step_index}."
                )

            current_t += (
                resolved_hold_durations_s[
                    step_index
                ]
            )

            current_speed = 0.0
            continue

        if isinstance(
            step,
            SemanticAccelerateToStepV2,
        ):
            if (
                step.target_speed_mps
                < current_speed - EPS
            ):
                raise ValueError(
                    "accelerate_to target is below "
                    "current speed."
                )

            duration = (
                step.target_speed_mps
                - current_speed
            ) / step.acceleration_mps2

            current_t += duration
            current_speed = step.target_speed_mps
            continue

        if isinstance(
            step,
            SemanticBrakeToStepV2,
        ):
            if (
                step.target_speed_mps
                > current_speed + EPS
            ):
                raise ValueError(
                    "brake_to target is above "
                    "current speed."
                )

            duration = (
                current_speed
                - step.target_speed_mps
            ) / step.deceleration_mps2

            current_t += duration
            current_speed = step.target_speed_mps
            continue

        raise TypeError(
            f"Unknown semantic behavior step: "
            f"{type(step)}"
        )

    return starts


def _collect_conditional_steps(
    semantic: SemanticScenarioV2,
):
    conditional_steps = []

    for actor in semantic.actors:

        for step_index, step in enumerate(
            actor.behavior.steps
        ):
            if isinstance(
                step,
                (
                    SemanticHoldUntilStepV2,
                    SemanticCruiseUntilStepV2,
                ),
            ):
                conditional_steps.append(
                    (
                        actor,
                        step_index,
                        step,
                    )
                )

    return conditional_steps


def compile_semantic_scenario_with_events(
    semantic: SemanticScenarioV2,
    max_iterations: int = 25,
):
    """
    Resolve all hold_until conditions OFFLINE.

    The returned ScenarioSpecV2 contains only deterministic physical
    motion. CARLA and HE never evaluate semantic conditions.

    Conditional durations are solved iteratively because one actor's
    event can depend on another actor whose own trajectory also
    contains conditional behavior.
    """
    conditional_steps = (
        _collect_conditional_steps(
            semantic
        )
    )

    # No conditional events -> normal compiler.
    if not conditional_steps:
        return compile_semantic_scenario(
            semantic
        )
    # --------------------------------------------------------
    # Initial provisional solution
    #
    # Every conditional hold initially lasts until approximately
    # scenario end. Later iterations shrink these durations.
    # --------------------------------------------------------

    hold_durations = {}
    cruise_durations = {}

    for actor, step_index, step in conditional_steps:

        if isinstance(
            step,
            SemanticHoldUntilStepV2,
        ):
            hold_durations.setdefault(
                actor.actor_id,
                {},
            )

            hold_durations[
                actor.actor_id
            ][step_index] = max(
                step.min_duration_s,
                semantic.duration_s,
            )

        elif isinstance(
            step,
            SemanticCruiseUntilStepV2,
        ):
            cruise_durations.setdefault(
                actor.actor_id,
                {},
            )

            cruise_durations[
                actor.actor_id
            ][step_index] = max(
                step.min_duration_s,
                semantic.duration_s,
            )

    frame_tolerance = (
        0.5 / semantic.fps
    )

    for iteration in range(
        max_iterations
    ):

        provisional_spec = (
            compile_semantic_scenario(
                semantic,
                resolved_hold_durations_s=(
                    hold_durations
                ),
                resolved_cruise_durations_s=(
                    cruise_durations
                ),
            )
        )

        provisional_resolved = (
            V2TrajectoryResolver().resolve(
                provisional_spec
            )
        )

        new_hold_durations = {
            actor_id: dict(values)
            for actor_id, values
            in hold_durations.items()
        }

        new_cruise_durations = {
            actor_id: dict(values)
            for actor_id, values
            in cruise_durations.items()
        }

        changed = False
        unresolved_this_iteration = False

        for (
            actor,
            step_index,
            step,
        ) in conditional_steps:

            local_starts = (
                _conditional_step_start_times(
                    actor.behavior,
                    hold_durations.get(
                        actor.actor_id,
                        {},
                    ),
                    cruise_durations.get(
                        actor.actor_id,
                        {},
                    ),
                )
            )

            hold_start_s = (
                actor.spawn_time_s
                + local_starts[
                    step_index
                ]
            )

            # A later conditional step may still lie beyond the
            # scenario horizon while an earlier hold is being solved.
            if (
                hold_start_s
                >= semantic.duration_s
            ):
                unresolved_this_iteration = True
                continue

            not_before_s = (
                hold_start_s
                + step.min_duration_s
            )

            try:
                event_time_s = (
                    resolve_condition_time(
                        provisional_resolved,
                        step.condition,
                        not_before_s=(
                            not_before_s
                        ),
                    )
                )

            except ValueError as exc:
                raise ValueError(
                    f"{semantic.scenario_id}: "
                    f"actor {actor.actor_id!r}, "
                    f"hold_until step {step_index}: "
                    f"{exc}"
                ) from exc

            duration_s = max(
                step.min_duration_s,
                event_time_s
                - hold_start_s,
            )

            if isinstance(
                step,
                SemanticHoldUntilStepV2,
            ):
                old_duration_s = (
                    hold_durations[
                        actor.actor_id
                    ][step_index]
                )

                new_hold_durations[
                    actor.actor_id
                ][step_index] = duration_s

            elif isinstance(
                step,
                SemanticCruiseUntilStepV2,
            ):
                old_duration_s = (
                    cruise_durations[
                        actor.actor_id
                    ][step_index]
                )

                new_cruise_durations[
                    actor.actor_id
                ][step_index] = duration_s

            else:
                raise TypeError(
                    f"Unknown conditional step: "
                    f"{type(step)}"
                )

            if (
                abs(
                    duration_s
                    - old_duration_s
                )
                > frame_tolerance
            ):
                changed = True

        hold_durations = new_hold_durations
        cruise_durations = new_cruise_durations

        if (
            not changed
            and not unresolved_this_iteration
        ):
            return compile_semantic_scenario(
                semantic,
                resolved_hold_durations_s=(
                    hold_durations
                ),
                resolved_cruise_durations_s=(
                    cruise_durations
                ),
            )

    raise RuntimeError(
        f"{semantic.scenario_id}: "
        "conditional event resolution did not "
        f"converge after {max_iterations} iterations. "
        "This may indicate a circular or impossible "
        "actor dependency."
    )