from __future__ import annotations

import math

from scenario_generator.schema import (
    ActorSpec,
    EgoMotionType,
    ManeuverType,
    ResolvedActorFrame,
    ResolvedActorInfo,
    ResolvedEgoFrame,
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

        ego_frames = self._resolve_ego_trajectory(
            scenario=scenario,
            total_frames=total_frames,
        )

        actor_frames: list[ResolvedActorFrame] = []

        for frame_idx in range(total_frames):
            t_s = frame_idx / scenario.fps

            for actor in scenario.actors:
                actor_frames.append(
                    self._resolve_actor_at_time(
                        actor=actor,
                        frame_idx=frame_idx,
                        t_s=t_s,
                    )
                )

        actor_info = [
            ResolvedActorInfo(
                actor_id=actor.actor_id,
                role=actor.role,
                actor_type=actor.actor_type,
                blueprint=actor.blueprint,
                dimensions_m=actor.dimensions_m,
            )
            for actor in scenario.actors
        ]

        return ResolvedScenario(
            scenario_id=scenario.scenario_id,
            source_description=scenario.description,
            duration_s=scenario.duration_s,
            fps=scenario.fps,
            coordinate_frame="ego_initial",
            camera=scenario.camera,
            camera_rig=scenario.camera_rig,
            road=scenario.road,
            actors=actor_info,
            ego_frames=ego_frames,
            frames=actor_frames,
        )
    def _resolve_ego_trajectory(
        self,
        scenario: ScenarioSpec,
        total_frames: int,
    ) -> list[ResolvedEgoFrame]:
        """
        Resolve the ego vehicle into the same ego-initial BEV frame
        used by all adversaries.

        This intentionally mirrors the deterministic integration style
        we validated with CARLA:
            state at frame
            -> update yaw
            -> move forward using updated yaw
        """

        ego = scenario.ego
        motion = ego.motion

        dt = 1.0 / float(scenario.fps)

        x_m = float(ego.initial_x_m)
        y_m = float(ego.initial_y_m)
        yaw_deg = float(ego.initial_yaw_deg)

        if motion.motion == EgoMotionType.RIGHT_TURN:
            total_yaw_change_deg = -abs(float(motion.turn_yaw_deg))

        elif motion.motion == EgoMotionType.LEFT_TURN:
            total_yaw_change_deg = abs(float(motion.turn_yaw_deg))

        else:
            total_yaw_change_deg = 0.0

        turn_start_frame = int(
            round(float(motion.turn_start_s) * scenario.fps)
        )

        turn_end_frame = int(
            round(
                (
                    float(motion.turn_start_s)
                    + float(motion.turn_duration_s)
                )
                * scenario.fps
            )
        )

        frames: list[ResolvedEgoFrame] = []

        for frame_idx in range(total_frames):
            t_s = frame_idx / float(scenario.fps)

            yaw_rad = math.radians(yaw_deg)
            
            if motion.motion == EgoMotionType.STATIC:
                speed_mps = 0.0
            else:
                speed_mps = float(ego.speed_mps)

            vx_mps = speed_mps * math.cos(yaw_rad)
            vy_mps = speed_mps * math.sin(yaw_rad)

            frames.append(
                ResolvedEgoFrame(
                    frame_idx=frame_idx,
                    t_s=t_s,
                    x_m=x_m,
                    y_m=y_m,
                    yaw_deg=yaw_deg,
                    speed_mps=speed_mps,
                    vx_mps=vx_mps,
                    vy_mps=vy_mps,
                )
            )

            if frame_idx >= total_frames - 1:
                continue

            if (
                motion.motion
                in (
                    EgoMotionType.LEFT_TURN,
                    EgoMotionType.RIGHT_TURN,
                )
                and turn_start_frame <= frame_idx < turn_end_frame
                and motion.turn_duration_s > 0.0
            ):
                yaw_rate_deg_per_s = (
                    total_yaw_change_deg
                    / float(motion.turn_duration_s)
                )
            else:
                yaw_rate_deg_per_s = 0.0

            # Update heading first.
            yaw_deg += yaw_rate_deg_per_s * dt

            # Then translate using the new heading.
            yaw_rad_next = math.radians(yaw_deg)

            x_m += (
                speed_mps
                * dt
                * math.cos(yaw_rad_next)
            )

            y_m += (
                speed_mps
                * dt
                * math.sin(yaw_rad_next)
            )

        return frames
    def _resolve_actor_at_time(
        self,
        actor: ActorSpec,
        frame_idx: int,
        t_s: float,
    ) -> ResolvedActorFrame:
        maneuver = actor.trajectory.maneuver

        if maneuver == ManeuverType.STATIC:
            x_m, y_m, yaw_deg, speed_mps, vx_mps, vy_mps = self._static(
                actor, t_s
            )

        elif maneuver == ManeuverType.ONCOMING:
            x_m, y_m, yaw_deg, speed_mps, vx_mps, vy_mps = self._oncoming(
                actor, t_s
            )

        elif maneuver == ManeuverType.CUT_IN:
            x_m, y_m, yaw_deg, speed_mps, vx_mps, vy_mps = self._cut_in(
                actor, t_s
            )

        elif maneuver == ManeuverType.FOLLOWING:
            x_m, y_m, yaw_deg, speed_mps, vx_mps, vy_mps = self._following(
                actor, t_s
            )

        elif maneuver == ManeuverType.CROSSING:
            x_m, y_m, yaw_deg, speed_mps, vx_mps, vy_mps = self._crossing(
                actor, t_s
            )

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
            vx_mps=vx_mps,
            vy_mps=vy_mps,
        )

    def _static(
        self,
        actor: ActorSpec,
        t_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        return (
            float(actor.initial_x_m),
            float(actor.initial_y_m),
            float(actor.initial_yaw_deg),
            0.0,   # speed
            0.0,   # vx
            0.0,   # vy
        )


    def _oncoming(
        self,
        actor: ActorSpec,
        t_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        speed = float(
            actor.trajectory.params.get(
                "speed_mps",
                actor.initial_speed_mps,
            )
        )

        target_y = float(
            actor.trajectory.params.get(
                "target_lane_y_m",
                actor.initial_y_m,
            )
        )

        # Oncoming vehicle travels toward ego along -x.
        x_m = float(actor.initial_x_m) - speed * t_s
        y_m = target_y

        yaw_deg = 180.0

        vx_mps = -speed
        vy_mps = 0.0

        return (
            x_m,
            y_m,
            yaw_deg,
            abs(speed),
            vx_mps,
            vy_mps,
        )


    def _cut_in(
        self,
        actor: ActorSpec,
        t_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Smooth lane-change trajectory.

        ScenarioGenerator convention:
            +x = forward
            +y = left
            +yaw = left

        Therefore:
            decreasing y -> moving right -> negative yaw
            increasing y -> moving left  -> positive yaw
        """

        longitudinal_speed = float(
            actor.trajectory.params.get(
                "speed_mps",
                actor.initial_speed_mps,
            )
        )

        target_y = float(
            actor.trajectory.params.get(
                "target_y_m",
                0.0,
            )
        )

        cut_start_s = float(
            actor.trajectory.params.get(
                "cut_start_s",
                1.5,
            )
        )

        cut_duration_s = max(
            1e-6,
            float(
                actor.trajectory.params.get(
                    "cut_duration_s",
                    3.0,
                )
            ),
        )

        x_m = (
            float(actor.initial_x_m)
            + longitudinal_speed * t_s
        )

        delta_y = target_y - float(actor.initial_y_m)

        # ------------------------------------------------------------
        # Before lane change
        # ------------------------------------------------------------
        if t_s <= cut_start_s:
            y_m = float(actor.initial_y_m)
            vy_mps = 0.0

        # ------------------------------------------------------------
        # After lane change
        # ------------------------------------------------------------
        elif t_s >= cut_start_s + cut_duration_s:
            y_m = target_y
            vy_mps = 0.0

        # ------------------------------------------------------------
        # During lane change
        # ------------------------------------------------------------
        else:
            u = (
                (t_s - cut_start_s)
                / cut_duration_s
            )

            s = smoothstep(u)

            y_m = (
                float(actor.initial_y_m)
                + delta_y * s
            )

            # Derivative of cubic smoothstep:
            #
            # s(u) = 3u^2 - 2u^3
            # ds/du = 6u(1-u)
            #
            ds_du = 6.0 * u * (1.0 - u)

            vy_mps = (
                delta_y
                * ds_du
                / cut_duration_s
            )

        vx_mps = longitudinal_speed

        # Heading follows the actual trajectory tangent.
        yaw_deg = math.degrees(
            math.atan2(
                vy_mps,
                vx_mps,
            )
        )

        total_speed = math.hypot(
            vx_mps,
            vy_mps,
        )

        return (
            x_m,
            y_m,
            yaw_deg,
            total_speed,
            vx_mps,
            vy_mps,
        )


    def _following(
        self,
        actor: ActorSpec,
        t_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        speed = float(
            actor.trajectory.params.get(
                "speed_mps",
                actor.initial_speed_mps,
            )
        )

        x_m = float(actor.initial_x_m) + speed * t_s
        y_m = float(actor.initial_y_m)

        return (
            x_m,
            y_m,
            0.0,
            abs(speed),
            speed,
            0.0,
        )


    def _crossing(
        self,
        actor: ActorSpec,
        t_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        speed = float(
            actor.trajectory.params.get(
                "speed_mps",
                actor.initial_speed_mps,
            )
        )

        target_x = float(
            actor.trajectory.params.get(
                "target_x_m",
                actor.initial_x_m,
            )
        )

        direction = float(
            actor.trajectory.params.get(
                "direction",
                -1.0,
            )
        )

        x_m = target_x
        y_m = (
            float(actor.initial_y_m)
            + direction * speed * t_s
        )

        vx_mps = 0.0
        vy_mps = direction * speed

        if vy_mps > 0.0:
            yaw_deg = 90.0
        elif vy_mps < 0.0:
            yaw_deg = -90.0
        else:
            yaw_deg = 0.0

        return (
            x_m,
            y_m,
            yaw_deg,
            abs(speed),
            vx_mps,
            vy_mps,
        )
