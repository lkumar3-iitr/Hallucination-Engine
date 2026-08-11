from __future__ import annotations

import math
from typing import List

import numpy as np

from scenario_generator.schema import (
    ActorSpec,
    ManeuverType,
    ResolvedActorFrame,
    ResolvedScenario,
    ScenarioSpec,
)


def smoothstep(u: float) -> float:
    """Cubic smoothstep from 0 to 1."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


class RuleBasedTrajectoryGenerator:
    """Rule-based trajectory generator v1.

    Coordinate convention:
    - x forward in meters.
    - y lateral in meters.
    - yaw_deg=0 means same direction as ego.
    - yaw_deg=180 means oncoming direction.
    """

    def resolve(self, scenario: ScenarioSpec) -> ResolvedScenario:
        total_frames = int(round(scenario.duration_s * scenario.fps)) + 1
        frames: list[ResolvedActorFrame] = []

        for frame_idx in range(total_frames):
            t_s = frame_idx / scenario.fps
            for actor in scenario.actors:
                frames.append(self._resolve_actor_at_time(actor, frame_idx, t_s))

        return ResolvedScenario(
            scenario_id=scenario.scenario_id,
            source_description=scenario.description,
            duration_s=scenario.duration_s,
            fps=scenario.fps,
            camera=scenario.camera,
            camera_rig=scenario.camera_rig,
            ego=scenario.ego,
            road=scenario.road,
            frames=frames,
        )

    def _resolve_actor_at_time(
        self,
        actor: ActorSpec,
        frame_idx: int,
        t_s: float,
    ) -> ResolvedActorFrame:
        maneuver = actor.trajectory.maneuver

        if maneuver == ManeuverType.STATIC:
            x_m, y_m, yaw_deg, speed_mps = self._static(actor, t_s)
        elif maneuver == ManeuverType.ONCOMING:
            x_m, y_m, yaw_deg, speed_mps = self._oncoming(actor, t_s)
        elif maneuver == ManeuverType.CUT_IN:
            x_m, y_m, yaw_deg, speed_mps = self._cut_in(actor, t_s)
        elif maneuver == ManeuverType.FOLLOWING:
            x_m, y_m, yaw_deg, speed_mps = self._following(actor, t_s)
        elif maneuver == ManeuverType.CROSSING:
            x_m, y_m, yaw_deg, speed_mps = self._crossing(actor, t_s)
        else:
            raise ValueError(f"Unsupported maneuver: {maneuver}")

        return ResolvedActorFrame(
            frame_idx=frame_idx,
            t_s=t_s,
            actor_id=actor.actor_id,
            x_m=x_m,
            y_m=y_m,
            yaw_deg=yaw_deg,
            speed_mps=speed_mps,
        )

    def _static(self, actor: ActorSpec, t_s: float) -> tuple[float, float, float, float]:
        return (
            actor.initial_x_m,
            actor.initial_y_m,
            actor.initial_yaw_deg,
            0.0,
        )

    def _oncoming(self, actor: ActorSpec, t_s: float) -> tuple[float, float, float, float]:
        speed_mps = float(actor.trajectory.params.get("speed_mps", actor.initial_speed_mps))
        target_lane_y_m = float(actor.trajectory.params.get("target_lane_y_m", actor.initial_y_m))

        # Oncoming means vehicle moves toward ego, so x decreases.
        x_m = actor.initial_x_m - speed_mps * t_s
        y_m = target_lane_y_m
        yaw_deg = 180.0
        return x_m, y_m, yaw_deg, speed_mps

    def _cut_in(self, actor: ActorSpec, t_s: float) -> tuple[float, float, float, float]:
        speed_mps = float(actor.trajectory.params.get("speed_mps", actor.initial_speed_mps))
        target_y_m = float(actor.trajectory.params.get("target_y_m", 0.0))
        cut_start_s = float(actor.trajectory.params.get("cut_start_s", 1.5))
        cut_duration_s = float(actor.trajectory.params.get("cut_duration_s", 3.0))

        x_m = actor.initial_x_m + speed_mps * t_s

        if t_s <= cut_start_s:
            y_m = actor.initial_y_m
            lateral_speed = 0.0
        elif t_s >= cut_start_s + cut_duration_s:
            y_m = target_y_m
            lateral_speed = 0.0
        else:
            u = (t_s - cut_start_s) / cut_duration_s
            s = smoothstep(u)
            y_m = actor.initial_y_m + (target_y_m - actor.initial_y_m) * s

            # Approximate lateral speed using derivative of smoothstep:
            ds_du = 6.0 * u * (1.0 - u)
            lateral_speed = abs((target_y_m - actor.initial_y_m) * ds_du / cut_duration_s)

        # Approximate yaw from lateral and longitudinal velocity.
        yaw_deg = math.degrees(math.atan2(target_y_m - actor.initial_y_m, max(speed_mps * cut_duration_s, 1e-6)))
        return x_m, y_m, yaw_deg, math.sqrt(speed_mps**2 + lateral_speed**2)

    def _following(self, actor: ActorSpec, t_s: float) -> tuple[float, float, float, float]:
        speed_mps = float(actor.trajectory.params.get("speed_mps", actor.initial_speed_mps))
        x_m = actor.initial_x_m + speed_mps * t_s
        return x_m, actor.initial_y_m, 0.0, speed_mps

    def _crossing(self, actor: ActorSpec, t_s: float) -> tuple[float, float, float, float]:
        speed_mps = float(actor.trajectory.params.get("speed_mps", actor.initial_speed_mps))
        target_x_m = float(actor.trajectory.params.get("target_x_m", actor.initial_x_m))
        direction = float(actor.trajectory.params.get("direction", -1.0))

        x_m = target_x_m
        y_m = actor.initial_y_m + direction * speed_mps * t_s
        yaw_deg = -90.0 if direction < 0 else 90.0
        return x_m, y_m, yaw_deg, speed_mps
