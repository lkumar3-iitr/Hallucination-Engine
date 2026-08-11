from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scenario_generator.schema import ResolvedScenario


def load_he_template(template_path: str | Path) -> dict[str, Any]:
    template_path = Path(template_path)
    with template_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_he_json(output_path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _frames_for_actor(resolved: ResolvedScenario, actor_id: str):
    frames = [f for f in resolved.frames if f.actor_id == actor_id]
    frames.sort(key=lambda f: f.frame_idx)
    return frames


def _estimate_velocity_from_resolved_frames(frames) -> dict[str, float]:
    """Estimate one constant HE velocity from resolved ScenarioGenerator frames.

    ScenarioGenerator convention:
        x_m = forward distance
        y_m = lateral distance

    Current HE compositor convention from the working scenario JSON:
        z_m = forward/depth distance
        x_m = lateral position
        y_m = vertical position

    Therefore:
        HE x_mps = ScenarioGenerator y velocity
        HE z_mps = ScenarioGenerator x velocity
    """

    if len(frames) < 2:
        return {
            "x_mps": 0.0,
            "y_mps": 0.0,
            "z_mps": 0.0,
        }

    first = frames[0]
    last = frames[-1]

    dt = last.t_s - first.t_s
    if dt <= 1e-6:
        return {
            "x_mps": 0.0,
            "y_mps": 0.0,
            "z_mps": 0.0,
        }

    scenario_forward_velocity_mps = (last.x_m - first.x_m) / dt
    scenario_lateral_velocity_mps = (last.y_m - first.y_m) / dt

    return {
        "x_mps": scenario_lateral_velocity_mps,
        "y_mps": 0.0,
        "z_mps": scenario_forward_velocity_mps,
    }


def _make_constant_velocity_adversary_from_resolved(
    resolved: ResolvedScenario,
    actor_id: str,
    template_adversary: dict[str, Any],
) -> dict[str, Any]:
    frames = _frames_for_actor(resolved, actor_id)
    if not frames:
        raise ValueError(f"No frames found for actor_id={actor_id}")

    first = frames[0]

    adversary = copy.deepcopy(template_adversary)

    adversary["id"] = actor_id
    adversary["enabled"] = True

    # ScenarioGenerator local state -> current HE state.
    adversary["initial_state"] = {
        "x_m": first.y_m,      # lateral
        "y_m": 0.0,            # vertical, flat road for now
        "z_m": first.x_m,      # forward/depth
        "yaw_deg": first.yaw_deg,
    }

    adversary["motion"] = {
        "model": "constant_velocity",
        "velocity": _estimate_velocity_from_resolved_frames(frames),
    }

    return adversary


def _make_keyframed_adversary_from_resolved(
    resolved: ResolvedScenario,
    actor_id: str,
    template_adversary: dict[str, Any],
) -> dict[str, Any]:
    """Future keyframed trajectory export.

    This is needed for cut-in, crossing, optimized, and learned trajectories.

    Important:
    Your current HE compositor may not support:
        motion.model == "keyframed_trajectory"

    So this export is for the next compositor update.
    """

    frames = _frames_for_actor(resolved, actor_id)
    if not frames:
        raise ValueError(f"No frames found for actor_id={actor_id}")

    first = frames[0]

    adversary = copy.deepcopy(template_adversary)

    adversary["id"] = actor_id
    adversary["enabled"] = True

    adversary["initial_state"] = {
        "x_m": first.y_m,
        "y_m": 0.0,
        "z_m": first.x_m,
        "yaw_deg": first.yaw_deg,
    }

    adversary["motion"] = {
        "model": "keyframed_trajectory",
        "coordinate_system": {
            "x_m": "lateral",
            "y_m": "vertical",
            "z_m": "forward_depth",
        },
        "keyframes": [
            {
                "frame_idx": f.frame_idx,
                "t_s": f.t_s,
                "x_m": f.y_m,
                "y_m": 0.0,
                "z_m": f.x_m,
                "yaw_deg": f.yaw_deg,
                "speed_mps": f.speed_mps,
            }
            for f in frames
        ],
    }
    # For keyframed trajectories such as cut-in / lane-change, sprite angle
    # should follow the trajectory tangent, not only the stored yaw.
    adversary.setdefault("rendering", {})
    adversary["rendering"]["angle_mode"] = "trajectory_tangent"
    adversary["rendering"]["angle_lateral_sign"] = -1.0
    adversary["rendering"]["angle_lateral_deadzone_ratio"] = 0.12
    adversary["rendering"]["angle_smoothing"] = False

    return adversary


def resolved_to_current_he_json(
    resolved: ResolvedScenario,
    template_json: dict[str, Any],
    output_dir: str | None = None,
    output_video_name: str | None = None,
    use_keyframes: bool = False,
) -> dict[str, Any]:
    """Convert ScenarioGenerator ResolvedScenario to current HE scenario JSON.

    Current safe mode:
        use_keyframes=False

    This works for:
        static
        oncoming
        approximate cut-in

    Future mode:
        use_keyframes=True

    This is better for:
        cut-in
        crossing
        optimization-based trajectory
        learning-based trajectory

    But the HE compositor must support keyframed_trajectory before rendering it.
    """

    he = copy.deepcopy(template_json)

    he["scenario_id"] = resolved.scenario_id
    he["description"] = resolved.source_description

    if output_dir is not None:
        he.setdefault("output", {})
        he["output"]["output_dir"] = output_dir

    if output_video_name is not None:
        he.setdefault("output", {})
        he["output"]["output_video_name"] = output_video_name

    # Timeline from resolved scenario.
    # Example: duration=8, fps=10 gives frames 0..80.
    end_frame = int(round(resolved.duration_s * resolved.fps))
    he["timeline"] = {
        "start_frame": 0,
        "end_frame": end_frame,
        "fps": float(resolved.fps),
    }

    # Ego motion from resolved scenario.
    he["ego_motion"] = {
        "enabled": True,
        "mode": "constant_forward_speed",
        "speed_mps": resolved.ego.speed_mps,
    }

    # Camera convention from resolved scenario.
    # Keep other template camera fields such as intrinsics, placement,
    # road_contact_model, and image_coordinate_system.
    he.setdefault("camera", {})
    he["camera"]["image_width"] = resolved.camera.image_width
    he["camera"]["image_height"] = resolved.camera.image_height
    he["camera"]["fov"] = resolved.camera.fov

    if "intrinsics" not in he["camera"]:
        he["camera"]["intrinsics"] = {
            "fx": 640.0,
            "fy": 640.0,
            "cx": resolved.camera.image_width / 2.0,
            "cy": resolved.camera.image_height / 2.0,
        }

    actor_ids = sorted({f.actor_id for f in resolved.frames})

    template_adversaries = he.get("adversaries", [])
    if template_adversaries:
        template_adv = template_adversaries[0]
    else:
        template_adv = {
            "id": "adv_vehicle_1",
            "type": "vehicle",
            "enabled": True,
            "size": {
                "length_m": 4.5,
                "width_m": 1.8,
                "height_m": 1.5,
            },
            "initial_state": {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 30.0,
                "yaw_deg": 0.0,
            },
            "motion": {
                "model": "constant_velocity",
                "velocity": {
                    "x_mps": 0.0,
                    "y_mps": 0.0,
                    "z_mps": 0.0,
                },
            },
            "rendering": {
                "alpha": 1.0,
                "angle_mode": "relative_yaw",
                "angle_smoothing": False,
                "shadow": False,
                "refiner": False,
            },
        }

    adversaries = []
    for actor_id in actor_ids:
        if use_keyframes:
            adversary = _make_keyframed_adversary_from_resolved(
                resolved=resolved,
                actor_id=actor_id,
                template_adversary=template_adv,
            )
        else:
            adversary = _make_constant_velocity_adversary_from_resolved(
                resolved=resolved,
                actor_id=actor_id,
                template_adversary=template_adv,
            )

        adversaries.append(adversary)

    he["adversaries"] = adversaries

    return he