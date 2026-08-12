from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scenario_generator.schema import ResolvedScenario


# ============================================================
# Basic JSON helpers
# ============================================================

def load_he_template(
    template_path: str | Path,
) -> dict[str, Any]:
    template_path = Path(template_path)

    with template_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_he_json(
    output_path: str | Path,
    data: dict[str, Any],
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )
        f.write("\n")


# ============================================================
# Coordinate conversion
# ============================================================

def sg_state_to_he_state(
    x_m: float,
    y_m: float,
    yaw_deg: float,
) -> dict[str, float]:
    """
    Convert ScenarioGenerator physical coordinates into the
    coordinate convention used by the current HE compositor.

    ScenarioGenerator:
        x = forward
        y = left
        +yaw = left turn

    HE:
        x = right
        y = vertical
        z = forward
        +yaw = right turn

    Therefore:

        HE x   = -SG y
        HE z   =  SG x
        HE yaw = -SG yaw
    """

    return {
        "x_m": -float(y_m),
        "y_m": 0.0,
        "z_m": float(x_m),
        "yaw_deg": -float(yaw_deg),
    }


# ============================================================
# Resolved-scenario helpers
# ============================================================

def _frames_for_actor(
    resolved: ResolvedScenario,
    actor_id: str,
):
    frames = [
        frame
        for frame in resolved.frames
        if frame.actor_id == actor_id
    ]

    frames.sort(
        key=lambda frame: frame.frame_idx
    )

    return frames


def _actor_info_by_id(
    resolved: ResolvedScenario,
):
    return {
        actor.actor_id: actor
        for actor in resolved.actors
    }


# ============================================================
# HE adversary generation
# ============================================================

def _make_keyframed_adversary_from_resolved(
    resolved: ResolvedScenario,
    actor_id: str,
    actor_info,
    template_adversary: dict[str, Any],
) -> dict[str, Any]:
    """
    Export one actor using the exact frame-by-frame resolved
    physical trajectory.

    No trajectory approximation is performed here.
    """

    frames = _frames_for_actor(
        resolved,
        actor_id,
    )

    if not frames:
        raise ValueError(
            f"No resolved frames found for actor_id={actor_id}"
        )

    info = actor_info[actor_id]

    adversary = copy.deepcopy(
        template_adversary
    )

    adversary["id"] = actor_id
    adversary["type"] = info.actor_type.value
    adversary["enabled"] = True

    # Keep source information for debugging / traceability.
    adversary["source"] = {
        "backend": "ScenarioGenerator",
        "actor_id": actor_id,
        "blueprint": info.blueprint,
        "coordinate_frame": resolved.coordinate_frame,
    }

    # --------------------------------------------------------
    # Physical dimensions
    # --------------------------------------------------------

    length_m, width_m, height_m = info.dimensions_m

    adversary["size"] = {
        "length_m": float(length_m),
        "width_m": float(width_m),
        "height_m": float(height_m),
    }

    # --------------------------------------------------------
    # Initial HE state
    # --------------------------------------------------------

    first = frames[0]

    first_he = sg_state_to_he_state(
        x_m=first.x_m,
        y_m=first.y_m,
        yaw_deg=first.yaw_deg,
    )

    adversary["initial_state"] = {
        "x_m": first_he["x_m"],
        "y_m": first_he["y_m"],
        "z_m": first_he["z_m"],
        "yaw_deg": first_he["yaw_deg"],
    }

    # --------------------------------------------------------
    # Exact resolved trajectory
    # --------------------------------------------------------

    keyframes = []

    for frame in frames:
        he_state = sg_state_to_he_state(
            x_m=frame.x_m,
            y_m=frame.y_m,
            yaw_deg=frame.yaw_deg,
        )

        keyframes.append(
            {
                "frame_idx": int(
                    frame.frame_idx
                ),
                "t_s": float(
                    frame.t_s
                ),

                "x_m": he_state["x_m"],
                "y_m": he_state["y_m"],
                "z_m": he_state["z_m"],
                "yaw_deg": he_state["yaw_deg"],

                # Not currently required by the compositor,
                # but useful metadata from the resolver.
                "speed_mps": float(
                    frame.speed_mps
                ),
            }
        )

    adversary["motion"] = {
        "model": "keyframed_trajectory",

        "coordinate_system": {
            "x_m": "lateral_right",
            "y_m": "vertical",
            "z_m": "forward",
            "yaw_positive": "right",
            "state_frame": "ego_initial",
        },

        "keyframes": keyframes,
    }

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    rendering = adversary.setdefault(
        "rendering",
        {},
    )

    rendering["alpha"] = float(
        rendering.get(
            "alpha",
            1.0,
        )
    )

    # Sprite selection is now based on the physical viewpoint:
    #
    # camera->actor bearing - actor relative yaw
    #
    # It must NOT be based purely on actor relative yaw or
    # trajectory tangent.
    rendering["angle_mode"] = "viewpoint"

    rendering["angle_smoothing"] = False

    rendering.setdefault(
        "shadow",
        False,
    )

    rendering.setdefault(
        "refiner",
        False,
    )

    # Remove obsolete angle-selection settings if inherited
    # from an old template.
    rendering.pop(
        "sprite_yaw_sign",
        None,
    )

    rendering.pop(
        "angle_lateral_sign",
        None,
    )

    rendering.pop(
        "angle_lateral_deadzone_ratio",
        None,
    )

    return adversary


# ============================================================
# Main adapter
# ============================================================

def resolved_to_current_he_json(
    resolved: ResolvedScenario,
    template_json: dict[str, Any],
    output_dir: str | None = None,
    output_video_name: str | None = None,
    use_keyframes: bool = True,
    background_video_path: str | None = None,
    ego_pose_path: str | None = None,
) -> dict[str, Any]:
    """
    Convert a backend-independent ResolvedScenario into a JSON
    directly consumable by run_he_temporal_compositor_v1.py.

    Design rule:

        ScenarioSpec
            ->
        ResolvedScenario
            ->
        exact keyframes
            ->
        HE compositor

    Actor states remain in the ego-initial physical frame until
    the HE compositor applies the recorded ego/camera trajectory.
    """

    if not use_keyframes:
        raise ValueError(
            "Constant-velocity export is deprecated. "
            "ResolvedScenario must be exported using exact "
            "keyframed trajectories."
        )

    he = copy.deepcopy(
        template_json
    )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    he["scenario_id"] = resolved.scenario_id

    he["description"] = (
        resolved.source_description
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    he.setdefault(
        "input",
        {},
    )

    if background_video_path is not None:
        he["input"]["video_path"] = (
            background_video_path
        )

    if ego_pose_path is not None:
        he["input"]["ego_pose_path"] = (
            ego_pose_path
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    he.setdefault(
        "output",
        {},
    )

    if output_dir is not None:
        he["output"]["output_dir"] = (
            output_dir
        )

    if output_video_name is not None:
        he["output"]["output_video_name"] = (
            output_video_name
        )

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------

    all_frame_indices = [
        frame.frame_idx
        for frame in resolved.ego_frames
    ]

    all_frame_indices.extend(
        frame.frame_idx
        for frame in resolved.frames
    )

    if not all_frame_indices:
        raise ValueError(
            "ResolvedScenario contains no frames."
        )

    end_frame = max(
        all_frame_indices
    )

    he["timeline"] = {
        "start_frame": 0,
        "end_frame": int(
            end_frame
        ),
        "fps": float(
            resolved.fps
        ),
    }

    # --------------------------------------------------------
    # Coordinate semantics
    # --------------------------------------------------------

    he["coordinate_mode"] = {
        "actor_state_frame": "ego_initial",
        "runtime_transform": "ego_pose_jsonl",
    }

    # The old synthetic ego_motion configuration must not be
    # used anymore. Ego motion comes from the real/background
    # ego pose sequence generated by the CARLA backend.
    he.pop(
        "ego_motion",
        None,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    he.setdefault(
        "camera",
        {},
    )

    he["camera"]["image_width"] = int(
        resolved.camera.image_width
    )

    he["camera"]["image_height"] = int(
        resolved.camera.image_height
    )

    he["camera"]["fov"] = float(
        resolved.camera.fov
    )

    # Preserve calibrated intrinsics from the working template
    # whenever available.
    if "intrinsics" not in he["camera"]:
        cx = (
            resolved.camera.image_width
            / 2.0
        )

        cy = (
            resolved.camera.image_height
            / 2.0
        )

        # For our canonical 1280x720 / 90-degree camera this is
        # 640 px. Keep the fallback explicit.
        fx = float(
            resolved.camera.image_width
            / 2.0
        )

        fy = fx

        he["camera"]["intrinsics"] = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    # --------------------------------------------------------
    # Visibility
    # --------------------------------------------------------

    he.setdefault(
        "visibility",
        {},
    )

    # We already decided that near-field clipping should be
    # handled generically later rather than culling at 2 m.
    he["visibility"]["min_render_depth_m"] = 0.0

    # --------------------------------------------------------
    # Adversaries
    # --------------------------------------------------------

    actor_info = _actor_info_by_id(
        resolved
    )

    template_adversaries = he.get(
        "adversaries",
        [],
    )

    if template_adversaries:
        template_adversary = (
            template_adversaries[0]
        )

    else:
        template_adversary = {
            "id": "adv_001",
            "type": "vehicle",
            "enabled": True,

            "size": {
                "length_m": 4.5,
                "width_m": 1.8,
                "height_m": 1.6,
            },

            "initial_state": {
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 30.0,
                "yaw_deg": 0.0,
            },

            "rendering": {
                "alpha": 1.0,
                "angle_mode": "viewpoint",
                "angle_smoothing": False,
                "shadow": False,
                "refiner": False,
            },
        }

    adversaries = []

    for actor in resolved.actors:
        adversaries.append(
            _make_keyframed_adversary_from_resolved(
                resolved=resolved,
                actor_id=actor.actor_id,
                actor_info=actor_info,
                template_adversary=template_adversary,
            )
        )

    he["adversaries"] = adversaries

    # --------------------------------------------------------
    # Traceability
    # --------------------------------------------------------

    he["scenario_generator_bridge"] = {
        "version": "v1",
        "source_coordinate_frame": (
            resolved.coordinate_frame
        ),
        "trajectory_export": (
            "exact_keyframed"
        ),

        "coordinate_conversion": {
            "he_x": "-scenario_y",
            "he_z": "scenario_x",
            "he_yaw": "-scenario_yaw",
        },

        "sprite_selection": "viewpoint",
    }

    return he