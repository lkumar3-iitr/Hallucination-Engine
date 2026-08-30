#!/usr/bin/env python3
"""
run_he_temporal_compositor_v1.py

Step 7 of HE temporal pipeline.

This script integrates:
  1. Scenario JSON loading
  2. Adversary state engine
  3. Approximate camera projection
  4. Relative angle computation
  5. Sprite selector
  6. Temporal compositor

Important:
  - This does NOT connect to CARLA.
  - This does NOT use CARLA world information.
  - It only uses input video + scenario JSON + sprite bank.

Example:
  python run_he_temporal_compositor_v1.py ^
    --scenario configs\\scenarios\\oncoming_vehicle_001.json ^
    --overwrite
"""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from heplacement_npz_lookup_adapter import HEPlacementNPZLookupAdapter
from he_geometry_calibration_v1 import HEGeometryCalibrationV1
import cv2
import numpy as np


# ============================================================
# Scenario utilities
# ============================================================

def try_import_v1_placement_adapter():
    from HallucinationEngine.models.HEPlacementModel.v1_mlp.he_runtime_adapter import HEPlacementRuntimeAdapter
    return HEPlacementRuntimeAdapter


def try_import_v2_lookup_placement():
    from HallucinationEngine.models.HEPlacementModel.v2_lookup.he_lookup_placement_v2 import HELookupPlacementV2
    return HELookupPlacementV2

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_frame_time(frame_idx, start_frame, fps):
    return (frame_idx - start_frame) / float(fps)


# ============================================================
# Ego pose utilities for ego-motion-aware rendering
# ============================================================

def load_ego_pose_jsonl(path):
    """
    Load ego pose records saved by record_carla_ego_route_video.py.

    Expected each line:
      {
        "recorded_frame_idx": 0,
        "t_s": 0.0,
        "ego_transform": {...},
        "camera_transform": {...}
      }

    Returns:
      dict[int, record], keyed by recorded_frame_idx
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ego pose JSONL not found: {path}")

    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rec = json.loads(line)
            idx = int(rec["recorded_frame_idx"])
            records[idx] = rec

    if not records:
        raise RuntimeError(f"No ego pose records loaded from: {path}")

    return records


def get_pose_record_for_frame(ego_pose_records, frame_idx):
    """
    Return ego pose record for current output video frame.

    If exact frame is missing, use nearest available frame.
    """

    if frame_idx in ego_pose_records:
        return ego_pose_records[frame_idx]

    keys = sorted(ego_pose_records.keys())

    if frame_idx <= keys[0]:
        return ego_pose_records[keys[0]]

    if frame_idx >= keys[-1]:
        return ego_pose_records[keys[-1]]

    best_key = min(keys, key=lambda k: abs(k - frame_idx))
    return ego_pose_records[best_key]


def yaw_to_forward_right_2d(yaw_deg):
    """
    Convert CARLA yaw angle to forward/right unit vectors in XY plane.

    CARLA/Unreal convention:
      X-Y is horizontal plane, Z is up.
      yaw rotates around Z.

    forward = [cos(yaw), sin(yaw)]
    right   = [-sin(yaw), cos(yaw)]
    """

    yaw = math.radians(float(yaw_deg))

    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)

    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)

    return (forward_x, forward_y), (right_x, right_y)


def normalize_angle_180_for_pose(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def actor_state_ego_initial_to_world(state, initial_pose_record):
    """
    Convert actor state from ego-initial local frame to CARLA world frame.

    Input state convention from current HE JSON:
      x_m = lateral
      y_m = vertical
      z_m = forward/depth
      yaw_deg = heading relative to ego-initial forward
    """

    ego0 = initial_pose_record["ego_transform"]

    ego0_x = float(ego0["x"])
    ego0_y = float(ego0["y"])
    ego0_z = float(ego0["z"])
    ego0_yaw = float(ego0["yaw"])

    forward, right = yaw_to_forward_right_2d(ego0_yaw)

    local_x = float(state["x_m"])
    local_y = float(state.get("y_m", 0.0))
    local_z = float(state["z_m"])

    world_x = ego0_x + forward[0] * local_z + right[0] * local_x
    world_y = ego0_y + forward[1] * local_z + right[1] * local_x
    world_z = ego0_z + local_y

    world_yaw = ego0_yaw + float(state.get("yaw_deg", 0.0))

    return {
        "world_x": float(world_x),
        "world_y": float(world_y),
        "world_z": float(world_z),
        "world_yaw_deg": float(world_yaw),
    }


def world_actor_to_current_camera_state(world_actor, current_pose_record):
    """
    Convert actor world pose to current camera-relative HE state.

    Output convention expected by HEPlacementModel/compositor:
      x_m = lateral in camera frame
      y_m = vertical in camera frame
      z_m = forward/depth in camera frame
      yaw_deg = actor heading relative to current camera yaw
    """

    camera_tf = current_pose_record["camera_transform"]

    cam_x = float(camera_tf["x"])
    cam_y = float(camera_tf["y"])
    cam_z = float(camera_tf["z"])
    cam_yaw = float(camera_tf["yaw"])

    forward, right = yaw_to_forward_right_2d(cam_yaw)

    dx = float(world_actor["world_x"]) - cam_x
    dy = float(world_actor["world_y"]) - cam_y
    dz = float(world_actor["world_z"]) - cam_z

    rel_z = dx * forward[0] + dy * forward[1]
    rel_x = dx * right[0] + dy * right[1]
    rel_y = dz

    rel_yaw = normalize_angle_180_for_pose(
        float(world_actor["world_yaw_deg"]) - cam_yaw
    )

    return {
        "x_m": float(rel_x),
        "y_m": float(rel_y),
        "z_m": float(rel_z),
        "yaw_deg": float(rel_yaw),
    }


def transform_state_with_ego_pose_if_enabled(
    state,
    scenario,
    ego_pose_records,
    initial_pose_record,
    frame_idx,
):
    """
    If coordinate_mode requests ego-pose transform, convert actor state from
    ego-initial frame to current camera-relative frame.

    Returns:
      transformed_state, transform_debug

    If transform is disabled:
      returns original state, minimal debug info.
    """

    coordinate_mode = scenario.get("coordinate_mode", {})
    actor_state_frame = coordinate_mode.get("actor_state_frame", "camera_relative")
    runtime_transform = coordinate_mode.get("runtime_transform", "none")

    transform_debug = {
        "enabled": False,
        "frame_idx": int(frame_idx),
        "actor_state_frame": actor_state_frame,
        "runtime_transform": runtime_transform,
        "raw_state": dict(state),
    }

    if runtime_transform != "ego_pose_jsonl":
        return state, transform_debug

    if actor_state_frame != "ego_initial":
        return state, transform_debug

    if ego_pose_records is None or initial_pose_record is None:
        raise RuntimeError(
            "coordinate_mode.runtime_transform is ego_pose_jsonl, "
            "but ego pose records are not loaded."
        )

    current_pose_record = get_pose_record_for_frame(
        ego_pose_records=ego_pose_records,
        frame_idx=frame_idx,
    )

    world_actor = actor_state_ego_initial_to_world(
        state=state,
        initial_pose_record=initial_pose_record,
    )

    transformed = world_actor_to_current_camera_state(
        world_actor=world_actor,
        current_pose_record=current_pose_record,
    )

    transformed["id"] = state["id"]
    transformed["type"] = state["type"]

    camera_tf = current_pose_record.get("camera_transform", {})
    ego_tf = current_pose_record.get("ego_transform", {})

    transform_debug = {
        "enabled": True,
        "frame_idx": int(frame_idx),
        "actor_state_frame": actor_state_frame,
        "runtime_transform": runtime_transform,

        # State before ego-pose transform.
        "raw_state": {
            "id": state.get("id"),
            "type": state.get("type"),
            "x_m": float(state.get("x_m", 0.0)),
            "y_m": float(state.get("y_m", 0.0)),
            "z_m": float(state.get("z_m", 0.0)),
            "yaw_deg": float(state.get("yaw_deg", 0.0)),
        },

        # Actor pose in CARLA/world frame after converting from ego-initial frame.
        "world_actor": {
            "world_x": float(world_actor["world_x"]),
            "world_y": float(world_actor["world_y"]),
            "world_z": float(world_actor["world_z"]),
            "world_yaw_deg": float(world_actor["world_yaw_deg"]),
        },

        # Current ego pose from JSONL.
        "ego_transform": {
            "x": float(ego_tf.get("x", 0.0)),
            "y": float(ego_tf.get("y", 0.0)),
            "z": float(ego_tf.get("z", 0.0)),
            "pitch": float(ego_tf.get("pitch", 0.0)),
            "yaw": float(ego_tf.get("yaw", 0.0)),
            "roll": float(ego_tf.get("roll", 0.0)),
        },

        # Current camera pose from JSONL.
        "camera_transform": {
            "x": float(camera_tf.get("x", 0.0)),
            "y": float(camera_tf.get("y", 0.0)),
            "z": float(camera_tf.get("z", 0.0)),
            "pitch": float(camera_tf.get("pitch", 0.0)),
            "yaw": float(camera_tf.get("yaw", 0.0)),
            "roll": float(camera_tf.get("roll", 0.0)),
        },

        # State after ego-pose transform; this is what placement sees.
        "transformed_state": {
            "id": transformed.get("id"),
            "type": transformed.get("type"),
            "x_m": float(transformed.get("x_m", 0.0)),
            "y_m": float(transformed.get("y_m", 0.0)),
            "z_m": float(transformed.get("z_m", 0.0)),
            "yaw_deg": float(transformed.get("yaw_deg", 0.0)),
        },
    }

    return transformed, transform_debug

def build_placement_adapter_if_enabled(scenario, frame_w, frame_h):
    placement_cfg = scenario.get("placement_model", {})

    if not placement_cfg.get("enabled", False):
        return None

    model_type = placement_cfg.get("type", "")

    # ------------------------------------------------------------
    # V1 learned MLP placement model
    # ------------------------------------------------------------
    if model_type == "heplacement_v1_mlp":
        checkpoint_path = placement_cfg["checkpoint_path"]
        device = placement_cfg.get("device", "cpu")

        camera = scenario["camera"]
        fov = float(camera.get("fov", 90.0))

        HEPlacementRuntimeAdapter = try_import_v1_placement_adapter()

        adapter = HEPlacementRuntimeAdapter(
            checkpoint_path=checkpoint_path,
            image_width=frame_w,
            image_height=frame_h,
            fov=fov,
            device=device,
        )

        print("[HEPlacement] Loaded V1 MLP placement adapter")
        print("[HEPlacement] checkpoint:", checkpoint_path)
        print("[HEPlacement] image size:", frame_w, "x", frame_h)
        print("[HEPlacement] fov       :", fov)
        print("[HEPlacement] device    :", device)

        return adapter

    # ------------------------------------------------------------
    # V2 deterministic lookup/interpolated placement model
    # ------------------------------------------------------------
    if model_type == "heplacement_v2_lookup":
        labels_path = placement_cfg["labels_path"]
        metadata_path = placement_cfg["metadata_path"]

        safe_distance_margin_m = float(
            placement_cfg.get("safe_distance_margin_m", 5.0)
        )

        default_mode = placement_cfg.get("mode", "clamp")

        HELookupPlacementV2 = try_import_v2_lookup_placement()

        adapter = HELookupPlacementV2(
            labels_path=labels_path,
            metadata_path=metadata_path,
            safe_distance_margin_m=safe_distance_margin_m,
            default_mode=default_mode,
        )

        print("[HEPlacement] Loaded V2 lookup placement adapter")
        print("[HEPlacement] labels     :", labels_path)
        print("[HEPlacement] metadata   :", metadata_path)
        print("[HEPlacement] method     :", placement_cfg.get("method", "interpolated"))
        print("[HEPlacement] mode       :", default_mode)
        print("[HEPlacement] safe min z :", adapter.safe_min_rel_z)

        return adapter

    raise ValueError(f"Unsupported placement model type: {model_type}")
# ============================================================
# Adversary state engine v1
# ============================================================

def update_constant_velocity_state(adversary, t_sec):
    init = adversary["initial_state"]
    vel = adversary["motion"]["velocity"]

    return {
        "id": adversary["id"],
        "type": adversary["type"],

        "x_m": float(init["x_m"]) + float(vel["x_mps"]) * t_sec,
        "y_m": float(init["y_m"]) + float(vel["y_mps"]) * t_sec,
        "z_m": float(init["z_m"]) + float(vel["z_mps"]) * t_sec,

        "yaw_deg": float(init["yaw_deg"])
    }

def lerp(a, b, t):
    return float(a) * (1.0 - t) + float(b) * t


def normalize_angle_180_local(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def lerp_angle_deg(a_deg, b_deg, t):
    """
    Interpolate angle along the shortest circular path.
    """
    a = float(a_deg)
    b = float(b_deg)
    delta = normalize_angle_180_local(b - a)
    return (a + delta * t) % 360.0


def update_linear_keyframe_state(adversary, t_norm):
    """
    t_norm is normalized timeline progress from 0 to 1.
    """
    start = adversary["motion"]["start"]
    end = adversary["motion"]["end"]

    t = max(0.0, min(1.0, float(t_norm)))

    return {
        "id": adversary["id"],
        "type": adversary["type"],

        "x_m": lerp(start["x_m"], end["x_m"], t),
        "y_m": lerp(start["y_m"], end["y_m"], t),
        "z_m": lerp(start["z_m"], end["z_m"], t),

        "yaw_deg": lerp_angle_deg(start["yaw_deg"], end["yaw_deg"], t)
    }


def update_keyframed_trajectory_state(adversary, frame_idx, t_sec):
    """
    Update state using motion.model == "keyframed_trajectory".

    Expected JSON format:

    "motion": {
      "model": "keyframed_trajectory",
      "keyframes": [
        {
          "frame_idx": 0,
          "t_s": 0.0,
          "x_m": 3.5,
          "y_m": 0.0,
          "z_m": 28.0,
          "yaw_deg": 0.0,
          "speed_mps": 4.0
        }
      ]
    }

    Interpolation priority:
      1. Use frame_idx if keyframes contain frame_idx.
      2. Otherwise use t_s.

    This lets ScenarioGenerator export exact cut-in / crossing / learned
    trajectories and the compositor follows them frame by frame.
    """

    motion = adversary["motion"]
    keyframes = motion.get("keyframes", [])

    if not keyframes:
        raise ValueError(
            f"Adversary {adversary.get('id')} has keyframed_trajectory but no keyframes"
        )

    # Sort once and cache inside adversary to avoid sorting every frame.
    if "_sorted_keyframes_cache" not in adversary:
        adversary["_sorted_keyframes_cache"] = sorted(
            keyframes,
            key=lambda k: (
                float(k.get("frame_idx", 0)),
                float(k.get("t_s", 0.0))
            )
        )

    kfs = adversary["_sorted_keyframes_cache"]

    # Decide whether this trajectory is frame-index based or time based.
    has_frame_idx = all("frame_idx" in k for k in kfs)

    if has_frame_idx:
        query_value = float(frame_idx)
        key_name = "frame_idx"
    else:
        query_value = float(t_sec)
        key_name = "t_s"

    # Before first keyframe: clamp to first.
    first = kfs[0]
    first_value = float(first.get(key_name, 0.0))
    if query_value <= first_value:
        return {
            "id": adversary["id"],
            "type": adversary["type"],
            "x_m": float(first["x_m"]),
            "y_m": float(first["y_m"]),
            "z_m": float(first["z_m"]),
            "yaw_deg": float(first["yaw_deg"]),
        }

    # After last keyframe: clamp to last.
    last = kfs[-1]
    last_value = float(last.get(key_name, 0.0))
    if query_value >= last_value:
        return {
            "id": adversary["id"],
            "type": adversary["type"],
            "x_m": float(last["x_m"]),
            "y_m": float(last["y_m"]),
            "z_m": float(last["z_m"]),
            "yaw_deg": float(last["yaw_deg"]),
        }

    # Interpolate between neighboring keyframes.
    for i in range(len(kfs) - 1):
        k0 = kfs[i]
        k1 = kfs[i + 1]

        v0 = float(k0.get(key_name, 0.0))
        v1 = float(k1.get(key_name, 0.0))

        if v0 <= query_value <= v1:
            denom = max(1e-6, v1 - v0)
            t = (query_value - v0) / denom

            return {
                "id": adversary["id"],
                "type": adversary["type"],

                "x_m": lerp(k0["x_m"], k1["x_m"], t),
                "y_m": lerp(k0["y_m"], k1["y_m"], t),
                "z_m": lerp(k0["z_m"], k1["z_m"], t),

                "yaw_deg": lerp_angle_deg(k0["yaw_deg"], k1["yaw_deg"], t),
            }

    # Safety fallback. Should not happen.
    return {
        "id": adversary["id"],
        "type": adversary["type"],
        "x_m": float(last["x_m"]),
        "y_m": float(last["y_m"]),
        "z_m": float(last["z_m"]),
        "yaw_deg": float(last["yaw_deg"]),
    }

def update_adversary_state(adversary, t_sec, t_norm, frame_idx=None):
    model = adversary["motion"]["model"]

    if model == "constant_velocity":
        return update_constant_velocity_state(adversary, t_sec)

    if model == "linear_keyframe":
        return update_linear_keyframe_state(adversary, t_norm)

    if model == "keyframed_trajectory":
        if frame_idx is None:
            raise ValueError("frame_idx is required for keyframed_trajectory")
        return update_keyframed_trajectory_state(
            adversary=adversary,
            frame_idx=frame_idx,
            t_sec=t_sec,
        )

    raise ValueError(f"Unsupported motion model: {model}")

# ============================================================
# Camera projection
# ============================================================

def project_point_pinhole(x_m, y_m, z_m, intrinsics):
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    if z_m <= 0.1:
        return None

    u = fx * (x_m / z_m) + cx
    v = fy * (y_m / z_m) + cy

    return u, v

def lerp_float(a, b, t):
    return float(a) * (1.0 - float(t)) + float(b) * float(t)


def interpolate_keypoints_by_z(z_m, keypoints):
    """
    Interpolate bottom_y from z_m using keypoints.

    keypoints example:
      [
        {"z_m": 45.0, "bottom_y": 255.0},
        {"z_m": 30.0, "bottom_y": 285.0},
        {"z_m": 20.0, "bottom_y": 330.0},
        {"z_m": 15.0, "bottom_y": 370.0}
      ]

    Works even if keypoints are not sorted.
    """

    if not keypoints:
        raise ValueError("road_contact_model.keypoints is empty")

    pts = sorted(
        [(float(p["z_m"]), float(p["bottom_y"])) for p in keypoints],
        key=lambda x: x[0]
    )

    z = float(z_m)

    # Clamp below nearest z.
    if z <= pts[0][0]:
        return pts[0][1]

    # Clamp above farthest z.
    if z >= pts[-1][0]:
        return pts[-1][1]

    # Interpolate between neighboring z points.
    for i in range(len(pts) - 1):
        z0, y0 = pts[i]
        z1, y1 = pts[i + 1]

        if z0 <= z <= z1:
            t = (z - z0) / max(1e-6, (z1 - z0))
            return lerp_float(y0, y1, t)

    return pts[-1][1]


def compute_bottom_y_from_road_contact_model(z_m, camera):
    """
    If camera.road_contact_model exists, use it.
    Otherwise return None and fall back to pinhole ground_y_m.
    """

    model = camera.get("road_contact_model", None)

    if model is None:
        return None

    model_type = model.get("type", "")

    if model_type == "z_to_bottom_y_keypoints":
        return interpolate_keypoints_by_z(
            z_m=z_m,
            keypoints=model["keypoints"]
        )

    raise ValueError(f"Unsupported road_contact_model type: {model_type}")

def project_vehicle_box(state, adversary, camera):
    image_width = int(camera["image_width"])
    image_height = int(camera["image_height"])
    intr = camera["intrinsics"]

    fx = float(intr["fx"])
    fy = float(intr["fy"])

    x = float(state["x_m"])
    z = float(state["z_m"])

    if z <= 0.1:
        return {
            "visible": False,
            "reason": "behind_camera_or_too_close"
        }

    size = adversary["size"]
    vehicle_width_m = float(size["width_m"])
    vehicle_height_m = float(size["height_m"])

    # Horizontal center still comes from pinhole projection.
    # For v1 road-contact correction:
    #   - x/z controls horizontal location
    #   - z controls size
    #   - road_contact_model controls bottom_y
    placement = camera.get("placement", {})
    ground_y_m = float(placement.get("ground_y_m", 1.35))

    center_bottom = project_point_pinhole(
        x_m=x,
        y_m=ground_y_m,
        z_m=z,
        intrinsics=intr
    )

    if center_bottom is None:
        return {
            "visible": False,
            "reason": "projection_failed"
        }

    u_center, pinhole_bottom_y = center_bottom

    calibrated_bottom_y = compute_bottom_y_from_road_contact_model(
        z_m=z,
        camera=camera
    )

    if calibrated_bottom_y is not None:
        v_bottom = calibrated_bottom_y
    else:
        v_bottom = pinhole_bottom_y

    box_w = fx * vehicle_width_m / z
    box_h = fy * vehicle_height_m / z

    x1 = u_center - box_w / 2.0
    x2 = u_center + box_w / 2.0
    y2 = v_bottom
    y1 = v_bottom - box_h

    margin = 100

    visible = not (
        x2 < -margin or
        x1 > image_width + margin or
        y2 < -margin or
        y1 > image_height + margin
    )

    return {
        "visible": bool(visible),

        "cx": float(u_center),
        "bottom_y": float(v_bottom),

        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),

        "box_width": float(box_w),
        "box_height": float(box_h),

        "z_m": float(z)
    }
def apply_placement_runtime_correction(rect, scenario, image_width, image_height):
    """
    Applies lightweight runtime calibration to HEPlacementModel output.

    This lets us correct small camera/perspective mismatches without retraining:
      - center_x offset/scale
      - bottom_y offset/scale around horizon
      - box width/height scale
    """

    cfg = scenario.get("placement_runtime_correction", {})

    if not cfg.get("enabled", False):
        return rect

    out = dict(rect)

    cx0 = float(image_width) / 2.0
    horizon_y = float(cfg.get("horizon_y_px", float(image_height) * 0.36))

    center_x = float(out["center_x"])
    bottom_y = float(out["bottom_y"])
    w = float(out["w"])
    h = float(out["h"])

    # Horizontal correction around image center.
    center_x = cx0 + (center_x - cx0) * float(
        cfg.get("center_x_scale_about_center", 1.0)
    )
    center_x += float(cfg.get("center_x_offset_px", 0.0))

    # Vertical correction around approximate horizon.
    bottom_y = horizon_y + (bottom_y - horizon_y) * float(
        cfg.get("bottom_y_scale_about_horizon", 1.0)
    )
    bottom_y += float(cfg.get("bottom_y_offset_px", 0.0))

    # Size correction.
    w *= float(cfg.get("box_w_scale", 1.0))
    h *= float(cfg.get("box_h_scale", 1.0))

    x = center_x - w / 2.0
    y = bottom_y - h

    out["x"] = float(x)
    out["y"] = float(y)
    out["w"] = float(w)
    out["h"] = float(h)
    out["center_x"] = float(center_x)
    out["bottom_y"] = float(bottom_y)

    return out

def project_vehicle_box_with_placement_model(
        state,
        adversary,
        placement_adapter,
        image_width,
        image_height,
        scenario=None
    ):
    """
    Convert placement model output to the same box format used by the compositor.

    Supports:
      1. New NPZ lookup adapter: predict_box(...)
      2. Old v2 lookup adapter: predict(...)
      3. Old v1 MLP adapter: predict_from_adversary_state(...)

    HEPlacement V2.1:
      - NPZ lookup provides bottom_y, width, height and visibility.
      - Horizontal center is computed analytically from camera intrinsics.
      - This avoids lateral clamping outside the lookup table rel_x range.
    """

    rel_x = float(state["x_m"])
    rel_z = float(state["z_m"])
    rel_yaw = float(state["yaw_deg"])

    # ------------------------------------------------------------
    # New NPZ lookup adapter
    # ------------------------------------------------------------
    if hasattr(placement_adapter, "predict_box"):

        box = placement_adapter.predict_box(
            state=state,
            image_width=image_width,
            image_height=image_height,
        )

        # ============================================================
        # HEPlacement V2.1
        #
        # Use analytic pinhole projection for horizontal center:
        #
        #     u = cx + fx * X / Z
        #
        # Preserve lookup-derived:
        #   - bottom_y
        #   - box width
        #   - box height
        #   - visibility
        #
        # This prevents lateral clamping when rel_x lies outside the
        # lookup table's supported lateral range.
        # ============================================================

        box = dict(box)

        camera_cfg = (
            scenario.get("camera", {})
            if scenario is not None
            else {}
        )

        intrinsics = camera_cfg.get(
            "intrinsics",
            {}
        )

        fx = float(
            intrinsics.get(
                "fx",
                image_width / 2.0,
            )
        )

        camera_cx = float(
            intrinsics.get(
                "cx",
                image_width / 2.0,
            )
        )

        if rel_z > 1e-6:

            analytic_cx = (
                camera_cx
                + fx * (rel_x / rel_z)
            )

            box_width = float(
                box["box_width"]
            )

            box["cx"] = float(
                analytic_cx
            )

            box["x1"] = float(
                analytic_cx
                - box_width / 2.0
            )

            box["x2"] = float(
                analytic_cx
                + box_width / 2.0
            )

            box["source"] = (
                "heplacement_v2_npz_lookup_analytic_cx"
            )

        # ------------------------------------------------------------
        # Debug output
        # ------------------------------------------------------------

        if not hasattr(
            project_vehicle_box_with_placement_model,
            "_debug_count"
        ):
            project_vehicle_box_with_placement_model._debug_count = 0

        if (
            project_vehicle_box_with_placement_model._debug_count
            < 20
        ):
            print(
                "[HEPlacement RAW: heplacement_v2_npz_lookup]"
            )

            print(
                "  input:",
                {
                    "rel_x": rel_x,
                    "rel_z": rel_z,
                    "rel_yaw": rel_yaw,
                }
            )

            print(
                "  box:",
                box
            )

            project_vehicle_box_with_placement_model._debug_count += 1

        # NPZ adapter already returns compositor box format.
        return box

    # ------------------------------------------------------------
    # Legacy placement adapters
    # ------------------------------------------------------------

    placement_cfg = (
        scenario.get(
            "placement_model",
            {}
        )
        if scenario is not None
        else {}
    )

    model_type = placement_cfg.get(
        "type",
        "heplacement_v1_mlp"
    )

    # ------------------------------------------------------------
    # Old V2 lookup adapter
    # ------------------------------------------------------------

    if model_type == "heplacement_v2_lookup":

        method = placement_cfg.get(
            "method",
            "interpolated"
        )

        mode = placement_cfg.get(
            "mode",
            "clamp"
        )

        rect = placement_adapter.predict(
            rel_x=rel_x,
            rel_z=rel_z,
            rel_yaw=rel_yaw,
            method=method,
            mode=mode,
        )

    # ------------------------------------------------------------
    # Old V1 MLP adapter
    # ------------------------------------------------------------

    elif model_type == "heplacement_v1_mlp":

        rect = placement_adapter.predict_from_adversary_state(
            {
                "rel_x": rel_x,
                "rel_z": rel_z,
                "rel_yaw": rel_yaw,
            }
        )

    else:

        raise ValueError(
            f"Unsupported placement model type: {model_type}"
        )

    # ------------------------------------------------------------
    # Optional legacy runtime correction
    # ------------------------------------------------------------

    if scenario is not None:

        rect = apply_placement_runtime_correction(
            rect=rect,
            scenario=scenario,
            image_width=image_width,
            image_height=image_height
        )

    # ------------------------------------------------------------
    # Debug output for legacy adapters
    # ------------------------------------------------------------

    if not hasattr(
        project_vehicle_box_with_placement_model,
        "_debug_count"
    ):
        project_vehicle_box_with_placement_model._debug_count = 0

    if (
        project_vehicle_box_with_placement_model._debug_count
        < 20
    ):

        print(
            f"[HEPlacement RAW: {model_type}]"
        )

        print(
            "  input:",
            {
                "rel_x": rel_x,
                "rel_z": rel_z,
                "rel_yaw": rel_yaw,
            }
        )

        print(
            "  rect:",
            rect
        )

        project_vehicle_box_with_placement_model._debug_count += 1

    # ------------------------------------------------------------
    # Convert legacy rectangle format to compositor box format
    # ------------------------------------------------------------

    x = float(rect["x"])
    y = float(rect["y"])

    w = float(rect["w"])
    h = float(rect["h"])

    center_x = float(
        rect["center_x"]
    )

    bottom_y = float(
        rect["bottom_y"]
    )

    visible = bool(
        rect["visible"]
    )

    visible_prob = float(
        rect.get(
            "visible_prob",
            1.0
        )
    )

    x1 = x
    y1 = y

    x2 = x + w
    y2 = y + h

    return {
        "visible": visible,
        "visible_prob": visible_prob,

        "cx": center_x,
        "bottom_y": bottom_y,

        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,

        "box_width": w,
        "box_height": h,

        "z_m": float(
            state["z_m"]
        ),

        "source": model_type
    }

def get_adversary_box(
        state,
        adversary,
        camera,
        image_width,
        image_height,
        placement_adapter=None,
        scenario=None
    ):
    if placement_adapter is not None:
        learned_box = project_vehicle_box_with_placement_model(
            state=state,
            adversary=adversary,
            placement_adapter=placement_adapter,
            image_width=image_width,
            image_height=image_height,
            scenario=scenario
        )

        if is_valid_predicted_box(learned_box, image_width, image_height):
            return learned_box

        print("[HEPlacement] Warning: invalid predicted box, falling back to geometric projection")

    fallback_box = project_vehicle_box(
        state=state,
        adversary=adversary,
        camera=camera
    )
    fallback_box["source"] = "geometric_projection"
    return fallback_box
# ============================================================
# Angle utilities + sprite selector
# ============================================================

def normalize_angle_360(angle_deg):
    return float(angle_deg) % 360.0


def normalize_angle_180(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def nearest_available_angle(angle_deg, available_angles):
    if not available_angles:
        raise ValueError("available_angles is empty")

    target = normalize_angle_360(angle_deg)

    best_angle = available_angles[0]
    best_dist = 999999.0

    for a in available_angles:
        d = abs(normalize_angle_180(target - a))
        if d < best_dist:
            best_dist = d
            best_angle = a

    return int(best_angle), float(best_dist)


def compute_relative_sprite_angle(state, camera_yaw_deg=0.0):
    yaw = float(state["yaw_deg"])
    rel = yaw - float(camera_yaw_deg)
    return normalize_angle_360(rel)

def compute_heading_yaw_from_velocity(
    vx_mps,
    vz_mps,
    fallback_yaw_deg=0.0,
    lateral_sign_for_angle=-1.0,
    lateral_deadzone_ratio=0.12,
):
    """
    Compute vehicle heading yaw from x-z velocity.

    HE coordinate convention:
      x_m = lateral image/world position
      z_m = forward/depth position

    Sprite convention:
      0 deg   = rear view
      90 deg  = one side view
      180 deg = front view
      270 deg = opposite side view

    Important:
    The placement x-axis and sprite-bank side convention may be mirrored.
    Therefore lateral_sign_for_angle is used only for sprite orientation.

    If the vehicle is mostly going straight away/toward camera, small lateral
    velocity should not cause side-view flicker, so we apply a deadzone.
    """

    vx = float(vx_mps)
    vz = float(vz_mps)

    if abs(vx) + abs(vz) < 1e-6:
        return normalize_angle_360(fallback_yaw_deg)

    # Remove tiny lateral drift when mostly straight.
    if abs(vz) > 1e-6:
        if abs(vx) / abs(vz) < float(lateral_deadzone_ratio):
            vx = 0.0

    # Flip lateral sign for sprite bank convention if needed.
    vx_for_angle = float(lateral_sign_for_angle) * vx

    yaw_deg = math.degrees(math.atan2(vx_for_angle, vz))
    return normalize_angle_360(yaw_deg)

def compute_keyframed_velocity_at_frame(adversary, frame_idx, t_sec):
    """
    Estimate local velocity for motion.model == keyframed_trajectory.

    Returns:
      vx_mps, vz_mps

    Uses neighboring keyframes. This is used only for sprite angle selection,
    not for position.
    """

    motion = adversary.get("motion", {})
    keyframes = motion.get("keyframes", [])

    if len(keyframes) < 2:
        return 0.0, 0.0

    if "_sorted_keyframes_cache" not in adversary:
        adversary["_sorted_keyframes_cache"] = sorted(
            keyframes,
            key=lambda k: (
                float(k.get("frame_idx", 0)),
                float(k.get("t_s", 0.0))
            )
        )

    kfs = adversary["_sorted_keyframes_cache"]
    has_frame_idx = all("frame_idx" in k for k in kfs)

    if has_frame_idx:
        query_value = float(frame_idx)
        key_name = "frame_idx"
    else:
        query_value = float(t_sec)
        key_name = "t_s"

    # Clamp before first segment.
    first_value = float(kfs[0].get(key_name, 0.0))
    if query_value <= first_value:
        k0 = kfs[0]
        k1 = kfs[1]
    # Clamp after last segment.
    elif query_value >= float(kfs[-1].get(key_name, 0.0)):
        k0 = kfs[-2]
        k1 = kfs[-1]
    else:
        k0 = kfs[0]
        k1 = kfs[1]

        for i in range(len(kfs) - 1):
            a = kfs[i]
            b = kfs[i + 1]

            va = float(a.get(key_name, 0.0))
            vb = float(b.get(key_name, 0.0))

            if va <= query_value <= vb:
                k0 = a
                k1 = b
                break

    dt = float(k1.get("t_s", 0.0)) - float(k0.get("t_s", 0.0))

    # If t_s is not useful, fall back to frame difference.
    if abs(dt) < 1e-6:
        f0 = float(k0.get("frame_idx", 0.0))
        f1 = float(k1.get("frame_idx", 0.0))
        df = max(1e-6, f1 - f0)

        # Approximate from frame index only. Absolute scale is not important
        # for yaw; direction is what matters.
        vx = (float(k1["x_m"]) - float(k0["x_m"])) / df
        vz = (float(k1["z_m"]) - float(k0["z_m"])) / df
        return vx, vz

    vx_mps = (float(k1["x_m"]) - float(k0["x_m"])) / dt
    vz_mps = (float(k1["z_m"]) - float(k0["z_m"])) / dt

    return vx_mps, vz_mps


def compute_constant_velocity_heading(adversary):
    """
    Estimate heading from constant_velocity motion.
    """

    motion = adversary.get("motion", {})
    velocity = motion.get("velocity", {})

    vx = float(velocity.get("x_mps", 0.0))
    vz = float(velocity.get("z_mps", 0.0))

    return vx, vz


def compute_viewpoint_sprite_angle(state):
    """
    Compute which physical side of the actor is visible from the camera.

    IMPORTANT:
    This is NOT simply the actor yaw relative to the camera.

    state is already in the current camera coordinate frame:

        x_m      = actor lateral position relative to camera
                   negative = left, positive = right

        z_m      = actor forward depth relative to camera

        yaw_deg  = actor heading relative to current camera heading

    Sprite-bank convention:

        0 deg   = rear
        90 deg  = one side
        180 deg = front
        270 deg = opposite side

    The camera-to-actor bearing is:

        bearing = atan2(x_m, z_m)

    The actual viewpoint around the vehicle is the bearing relative
    to the actor's own heading:

        sprite_angle = bearing - actor_relative_yaw

    This has an important property:

    If the camera rotates in place, both bearing and actor-relative
    yaw change by the same amount, so the selected sprite does NOT
    change.

    The sprite changes only when the physical viewpoint around the
    actor changes because:
        - actor rotates,
        - actor translates,
        - camera/ego translates,
        - or a combination of these.
    """

    x_m = float(state.get("x_m", 0.0))
    z_m = float(state.get("z_m", 0.0))
    actor_relative_yaw_deg = float(state.get("yaw_deg", 0.0))

    # Direction from the camera toward the actor, expressed in the
    # current camera horizontal coordinate system.
    bearing_deg = math.degrees(
        math.atan2(x_m, z_m)
    )

    viewpoint_angle_deg = (
        bearing_deg - actor_relative_yaw_deg
    )

    return normalize_angle_360(viewpoint_angle_deg)


def compute_sprite_angle_for_adversary(
    state,
    adversary,
    camera_yaw_deg=0.0,
    frame_idx=None,
    t_sec=0.0,
):
    """
    Select the sprite from the physical camera-to-actor viewpoint.

    camera_yaw_deg, frame_idx and t_sec are retained in the signature
    for compatibility with the existing compositor call site.

    Actor orientation and camera rotation have already been accounted
    for in `state`, so camera yaw must not be applied again here.
    """

    return compute_viewpoint_sprite_angle(state)

# ============================================================
# View-matrix sprite selection
# ============================================================

def sprite_filename_from_any_path(path_text):
    """
    Extract only the filename from either a Windows or Linux path.

    This allows sprite-bank CSV files generated on the server to be
    copied to Windows without rewriting their stored absolute paths.
    """

    text = str(path_text).replace("\\", "/")
    return text.split("/")[-1]


def make_view_matrix_key(
    angle_deg,
    distance_m,
    elevation_deg,
):
    return (
        int(angle_deg) % 360,
        round(float(distance_m), 6),
        round(float(elevation_deg), 6),
    )


def circular_angle_error_deg(a_deg, b_deg):
    return abs(
        normalize_angle_180(
            float(a_deg) - float(b_deg)
        )
    )


def nearest_linear_value(
    query_value,
    available_values,
):
    return min(
        available_values,
        key=lambda value: abs(
            float(query_value) - float(value)
        ),
    )

def nearest_view_distance(
    query_distance_m,
    available_distances,
    mode="linear",
):
    """
    Choose a captured view-matrix distance.

    Modes
    -----
    linear:
        nearest physical distance |d - d_i|

    log:
        nearest multiplicative distance in log(d)

    inverse_depth:
        nearest perspective coordinate |1/d - 1/d_i|

    The default remains 'linear' for backwards compatibility.
    """

    query_distance_m = float(
        query_distance_m
    )

    distances = [
        float(v)
        for v in available_distances
    ]

    if not distances:
        raise ValueError(
            "available_distances is empty"
        )

    if mode == "linear":

        return min(
            distances,
            key=lambda value:
                abs(
                    query_distance_m
                    - value
                ),
        )

    if mode == "log":

        if query_distance_m <= 0:
            raise ValueError(
                "log distance selection requires "
                "query_distance_m > 0"
            )

        return min(
            distances,
            key=lambda value:
                abs(
                    math.log(
                        query_distance_m
                    )
                    -
                    math.log(
                        value
                    )
                ),
        )

    if mode == "inverse_depth":

        if query_distance_m <= 0:
            raise ValueError(
                "inverse-depth selection requires "
                "query_distance_m > 0"
            )

        query_inverse_depth = (
            1.0
            / query_distance_m
        )

        return min(
            distances,
            key=lambda value:
                abs(
                    query_inverse_depth
                    -
                    (
                        1.0
                        / value
                    )
                ),
        )

    raise ValueError(
        "Unsupported distance_selection_mode: "
        f"{mode}"
    )
def csv_optional_float(
    row,
    key,
    default=None,
):
    value = row.get(
        key,
        None,
    )

    if value is None:
        return default

    text = str(
        value
    ).strip()

    if not text:
        return default

    try:
        return float(
            text
        )
    except Exception:
        return default


def csv_optional_int(
    row,
    key,
    default=None,
):
    value = csv_optional_float(
        row=row,
        key=key,
        default=None,
    )

    if value is None:
        return default

    try:
        return int(
            round(
                value
            )
        )
    except Exception:
        return default


def csv_optional_bool(
    row,
    key,
    default=None,
):
    value = row.get(
        key,
        None,
    )

    if value is None:
        return default

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
    }:
        return False

    return default


def csv_optional_text(
    row,
    key,
    default="",
):
    value = row.get(
        key,
        None,
    )

    if value is None:
        return default

    return str(
        value
    )
def load_view_matrix_sprite_bank(sprite_bank):
    """
    Load one or more generated view_matrix.csv files.

    Expected sprite_bank configuration:

        {
            "mode": "view_matrix",

            "view_matrix_csvs": [
                "D:/HE_Data/.../e0/view_matrix.csv",
                "D:/HE_Data/.../e10/view_matrix.csv"
            ],

            "target_height_m": 0.75
        }

    Returns a dictionary containing:

        records
        index
        angles
        distances
        elevations
    """

    csv_paths = sprite_bank.get(
        "view_matrix_csvs",
        []
    )

    if not csv_paths:
        raise RuntimeError(
            "sprite_bank.mode='view_matrix' but "
            "sprite_bank.view_matrix_csvs is empty."
        )

    records = []
    index = {}

    asset_metadata = None

    for csv_path_text in csv_paths:

        csv_path = Path(
            csv_path_text
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"View-matrix CSV not found: {csv_path}"
            )

        bank_root = csv_path.parent
        rgba_dir = bank_root / "rgba"
        asset_metadata_path = (
            bank_root
            /
            "asset_metadata.json"
        )

        if asset_metadata_path.exists():

            with open(
                asset_metadata_path,
                "r",
                encoding="utf-8",
            ) as metadata_file:

                current_asset_metadata = (
                    json.load(
                        metadata_file
                    )
                )

            if asset_metadata is None:

                asset_metadata = (
                    current_asset_metadata
                )

            else:

                existing_asset_id = (
                    asset_metadata.get(
                        "asset_id"
                    )
                )

                current_asset_id = (
                    current_asset_metadata.get(
                        "asset_id"
                    )
                )

                if (
                    existing_asset_id
                    !=
                    current_asset_id
                ):
                    raise RuntimeError(
                        "View-matrix CSV files belong "
                        "to different assets: "
                        f"{existing_asset_id!r} vs "
                        f"{current_asset_id!r}"
                    )
        local_count = 0

        with open(
            csv_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                angle_deg = int(
                    float(
                        row["angle_deg"]
                    )
                ) % 360

                distance_m = float(
                    row["distance_m"]
                )

                elevation_deg = float(
                    row["elevation_deg"]
                )

                rgba_filename = (
                    sprite_filename_from_any_path(
                        row.get(
                            "rgba_path",
                            ""
                        )
                    )
                )

                local_rgba_path = (
                    rgba_dir
                    /
                    rgba_filename
                )

                key = make_view_matrix_key(
                    angle_deg,
                    distance_m,
                    elevation_deg,
                )

                crop_x1_px = csv_optional_float(
                    row,
                    "crop_x1_px",
                )

                crop_y1_px = csv_optional_float(
                    row,
                    "crop_y1_px",
                )

                projected_ground_anchor_x_px = (
                    csv_optional_float(
                        row,
                        "projected_ground_anchor_x_px",
                    )
                )

                projected_ground_anchor_y_px = (
                    csv_optional_float(
                        row,
                        "projected_ground_anchor_y_px",
                    )
                )

                # Physical support point expressed in the selected
                # cropped sprite's own pixel coordinates.
                #
                # IMPORTANT:
                # This is metadata only for now. The current renderer
                # will continue to use anchor_x / anchor_y until the
                # physical-anchor rendering change is validated.
                source_ground_anchor_x = None
                source_ground_anchor_y = None

                if (
                    projected_ground_anchor_x_px is not None
                    and
                    crop_x1_px is not None
                ):
                    source_ground_anchor_x = (
                        projected_ground_anchor_x_px
                        -
                        crop_x1_px
                    )

                if (
                    projected_ground_anchor_y_px is not None
                    and
                    crop_y1_px is not None
                ):
                    source_ground_anchor_y = (
                        projected_ground_anchor_y_px
                        -
                        crop_y1_px
                    )

                record = {
                    # =================================================
                    # View selector
                    # =================================================

                    "angle_deg":
                        angle_deg,

                    "distance_m":
                        distance_m,

                    "elevation_deg":
                        elevation_deg,

                    # =================================================
                    # Asset identity / schema
                    # =================================================

                    "schema_version":
                        csv_optional_text(
                            row,
                            "schema_version",
                        ),

                    "generation_status":
                        csv_optional_text(
                            row,
                            "generation_status",
                        ),

                    "asset_id":
                        csv_optional_text(
                            row,
                            "asset_id",
                        ),

                    "asset_class":
                        csv_optional_text(
                            row,
                            "asset_class",
                        ),

                    "carla_blueprint":
                        csv_optional_text(
                            row,
                            "carla_blueprint",
                        ),

                    # =================================================
                    # Source sprite
                    # =================================================

                    "sprite_width_px":
                        csv_optional_int(
                            row,
                            "sprite_width_px",
                            0,
                        ),

                    "sprite_height_px":
                        csv_optional_int(
                            row,
                            "sprite_height_px",
                            0,
                        ),

                    # Legacy alpha-bottom anchor.
                    #
                    # KEEP for now so rendering behavior is unchanged.
                    "anchor_x":
                        csv_optional_float(
                            row,
                            "anchor_x",
                            0.0,
                        ),

                    "anchor_y":
                        csv_optional_float(
                            row,
                            "anchor_y",
                            0.0,
                        ),

                    "rgba_path":
                        str(
                            local_rgba_path
                        ),

                    # =================================================
                    # Capture camera
                    # =================================================

                    "target_height_m":
                        csv_optional_float(
                            row,
                            "target_height_m",
                        ),

                    "image_width_px":
                        csv_optional_int(
                            row,
                            "image_width_px",
                        ),

                    "image_height_px":
                        csv_optional_int(
                            row,
                            "image_height_px",
                        ),

                    "fov_deg":
                        csv_optional_float(
                            row,
                            "fov_deg",
                        ),

                    "camera_fx_px":
                        csv_optional_float(
                            row,
                            "camera_fx_px",
                        ),

                    "camera_fy_px":
                        csv_optional_float(
                            row,
                            "camera_fy_px",
                        ),

                    "camera_cx_px":
                        csv_optional_float(
                            row,
                            "camera_cx_px",
                        ),

                    "camera_cy_px":
                        csv_optional_float(
                            row,
                            "camera_cy_px",
                        ),

                    # =================================================
                    # Physical 3-D / camera geometry
                    # =================================================

                    "bbox_center_distance_m":
                        csv_optional_float(
                            row,
                            "bbox_center_distance_m",
                        ),

                    "actor_depth_m":
                        csv_optional_float(
                            row,
                            "actor_depth_m",
                        ),

                    "camera_forward_distance_m":
                        csv_optional_float(
                            row,
                            "camera_forward_distance_m",
                        ),

                    "camera_right_offset_m":
                        csv_optional_float(
                            row,
                            "camera_right_offset_m",
                        ),

                    "camera_up_offset_m":
                        csv_optional_float(
                            row,
                            "camera_up_offset_m",
                        ),

                    "support_depth_m":
                        csv_optional_float(
                            row,
                            "support_depth_m",
                        ),

                    "nearest_bbox_depth_m":
                        csv_optional_float(
                            row,
                            "nearest_bbox_depth_m",
                        ),

                    "farthest_bbox_depth_m":
                        csv_optional_float(
                            row,
                            "farthest_bbox_depth_m",
                        ),

                    # =================================================
                    # Projected physical bbox
                    # =================================================

                    "projected_bbox_x1_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_x1_px",
                        ),

                    "projected_bbox_y1_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_y1_px",
                        ),

                    "projected_bbox_x2_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_x2_px",
                        ),

                    "projected_bbox_y2_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_y2_px",
                        ),

                    "projected_bbox_width_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_width_px",
                        ),

                    "projected_bbox_height_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_height_px",
                        ),

                    "projected_bbox_center_x_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_center_x_px",
                        ),

                    "projected_bbox_bottom_y_px":
                        csv_optional_float(
                            row,
                            "projected_bbox_bottom_y_px",
                        ),

                    # =================================================
                    # Physical support / ground anchor
                    # =================================================

                    "projected_ground_anchor_x_px":
                        projected_ground_anchor_x_px,

                    "projected_ground_anchor_y_px":
                        projected_ground_anchor_y_px,

                    "source_ground_anchor_x":
                        source_ground_anchor_x,

                    "source_ground_anchor_y":
                        source_ground_anchor_y,

                    # =================================================
                    # Crop geometry
                    # =================================================

                    "crop_x1_px":
                        crop_x1_px,

                    "crop_y1_px":
                        crop_y1_px,

                    "crop_x2_px":
                        csv_optional_float(
                            row,
                            "crop_x2_px",
                        ),

                    "crop_y2_px":
                        csv_optional_float(
                            row,
                            "crop_y2_px",
                        ),

                    "crop_width_px":
                        csv_optional_float(
                            row,
                            "crop_width_px",
                        ),

                    "crop_height_px":
                        csv_optional_float(
                            row,
                            "crop_height_px",
                        ),

                    # =================================================
                    # Visible alpha geometry
                    # =================================================

                    "alpha_area_px":
                        csv_optional_int(
                            row,
                            "alpha_area_px",
                        ),

                    "visible_width_px":
                        csv_optional_float(
                            row,
                            "visible_width_px",
                        ),

                    "visible_height_px":
                        csv_optional_float(
                            row,
                            "visible_height_px",
                        ),

                    "connected_component_count":
                        csv_optional_int(
                            row,
                            "connected_component_count",
                        ),

                    # =================================================
                    # HE physical size references
                    # =================================================

                    "he_reference_projected_width_px":
                        csv_optional_float(
                            row,
                            "he_reference_projected_width_px",
                        ),

                    "he_reference_projected_height_px":
                        csv_optional_float(
                            row,
                            "he_reference_projected_height_px",
                        ),

                    # =================================================
                    # QA
                    # =================================================

                    "qa_pass":
                        csv_optional_bool(
                            row,
                            "qa_pass",
                        ),

                    "qa_flags":
                        csv_optional_text(
                            row,
                            "qa_flags",
                        ),

                    "qa_warnings":
                        csv_optional_text(
                            row,
                            "qa_warnings",
                        ),
                }

                if key in index:
                    raise RuntimeError(
                        f"Duplicate view-matrix key: {key}"
                    )

                if not local_rgba_path.exists():
                    raise FileNotFoundError(
                        f"View-matrix sprite not found: "
                        f"{local_rgba_path}"
                    )

                index[key] = record
                records.append(record)
                local_count += 1

        print(
            "[ViewMatrix] Loaded",
            local_count,
            "views from",
            csv_path,
        )

    angles = sorted(
        set(
            int(record["angle_deg"])
            for record in records
        )
    )

    distances = sorted(
        set(
            float(record["distance_m"])
            for record in records
        )
    )

    elevations = sorted(
        set(
            float(record["elevation_deg"])
            for record in records
        )
    )

    expected_count = (
        len(angles)
        *
        len(distances)
        *
        len(elevations)
    )

    print(
        "[ViewMatrix] Total views:",
        len(records)
    )

    print(
        "[ViewMatrix] Angles:",
        len(angles)
    )

    print(
        "[ViewMatrix] Distances:",
        distances
    )

    print(
        "[ViewMatrix] Elevations:",
        elevations
    )

    if len(index) != expected_count:
        print(
            "[ViewMatrix] WARNING: bank is not a complete "
            "angle x distance x elevation grid."
        )

    return {
        "records": records,
        "index": index,
        "angles": angles,
        "distances": distances,
        "elevations": elevations,

        "asset_metadata":
            asset_metadata,

        "physical_bbox":
            (
                asset_metadata.get(
                    "physical_bbox"
                )
                if asset_metadata is not None
                else None
            ),

        "capture_metadata":
            (
                asset_metadata.get(
                    "capture"
                )
                if asset_metadata is not None
                else None
            ),
    }


def compute_view_matrix_coordinates(
    state,
    target_height_m=0.75,
    vertical_mode="state_y",
    camera_height_m=1.60,
):
    """
    Convert final HE camera-relative state into view-matrix coordinates.

    HE state:
        x_m = lateral
        y_m = actor-base vertical position relative to camera
        z_m = forward depth
        yaw_deg = actor heading relative to camera

    Returns:
        viewpoint_angle_deg
        distance_m
        elevation_deg
    """

    x_m = float(
        state.get(
            "x_m",
            0.0
        )
    )

    y_m = float(
        state.get(
            "y_m",
            0.0
        )
    )

    z_m = float(
        state.get(
            "z_m",
            0.0
        )
    )

    yaw_deg = float(
        state.get(
            "yaw_deg",
            0.0
        )
    )

    # Production 4320 sprite-bank convention:
    #
    #   0 deg   = FRONT view
    #   180 deg = REAR view
    #
    # The older HE viewpoint helper uses the opposite convention,
    # so rotate only the production view-matrix query by 180 deg.
    viewpoint_angle_deg = normalize_angle_360(
        compute_viewpoint_sprite_angle(
            state
        )
        + 180.0
    )

    if vertical_mode == "state_y":

        # y_m is already the actor base height relative to camera.
        # This is correct for ego-pose/world transformed states.
        target_y_m = (
            y_m
            +
            float(target_height_m)
        )

    elif vertical_mode == "level_ground":

        # Compatibility mode for older HE scenarios where y_m=0
        # means "actor is on the road", rather than literally at
        # the camera's vertical position.
        target_y_m = (
            -float(camera_height_m)
            +
            float(target_height_m)
        )

    else:

        raise ValueError(
            f"Unsupported view-matrix vertical_mode: "
            f"{vertical_mode}"
        )

    horizontal_distance_m = math.sqrt(
        x_m * x_m
        +
        z_m * z_m
    )

    distance_m = math.sqrt(
        horizontal_distance_m
        *
        horizontal_distance_m
        +
        target_y_m
        *
        target_y_m
    )

    elevation_deg = math.degrees(
        math.atan2(
            -target_y_m,
            horizontal_distance_m,
        )
    )

    return {
        "viewpoint_angle_deg":
            float(viewpoint_angle_deg),

        "distance_m":
            float(distance_m),

        "elevation_deg":
            float(elevation_deg),

        "horizontal_distance_m":
            float(horizontal_distance_m),

        "target_y_m":
            float(target_y_m),

        "yaw_deg":
            float(yaw_deg),
    }


def select_view_matrix_sprite(
    state,
    sprite_bank,
    view_matrix,
):
    """
    Select nearest available:

        azimuth
        distance
        elevation
    """

    asset_metadata = (
        view_matrix.get(
            "asset_metadata"
        )
        or
        {}
    )

    capture_metadata = (
        asset_metadata.get(
            "capture"
        )
        or
        {}
    )

    target_height_m = float(
        capture_metadata.get(
            "resolved_target_height_m",
            sprite_bank.get(
                "target_height_m",
                0.75,
            ),
        )
    )

    vertical_mode = sprite_bank.get(
        "vertical_mode",
        "state_y",
    )

    camera_height_m = float(
        sprite_bank.get(
            "camera_height_m",
            1.60,
        )
    )

    query = compute_view_matrix_coordinates(
        state=state,
        target_height_m=target_height_m,
        vertical_mode=vertical_mode,
        camera_height_m=camera_height_m,
    )

    selected_angle = min(
        view_matrix["angles"],
        key=lambda value:
            circular_angle_error_deg(
                query[
                    "viewpoint_angle_deg"
                ],
                value,
            ),
    )

    distance_selection_mode = (
        sprite_bank.get(
            "distance_selection_mode",
            "linear",
        )
    )

    selected_distance = (
        nearest_view_distance(
            query_distance_m=
                query["distance_m"],

            available_distances=
                view_matrix["distances"],

            mode=
                distance_selection_mode,
        )
    )

    selected_elevation = nearest_linear_value(
        query["elevation_deg"],
        view_matrix["elevations"],
    )

    key = make_view_matrix_key(
        selected_angle,
        selected_distance,
        selected_elevation,
    )

    record = (
        view_matrix["index"].get(
            key
        )
    )

    if record is None:
        raise RuntimeError(
            f"Selected view-matrix coordinate "
            f"does not exist: {key}"
        )

    sprite_path = Path(
        record["rgba_path"]
    )

    return {
        "mode": "view_matrix",

        "relative_angle_deg":
            float(
                query[
                    "viewpoint_angle_deg"
                ]
            ),

        "selected_angle":
            int(selected_angle),

        "angle_error_deg":
            float(
                circular_angle_error_deg(
                    query[
                        "viewpoint_angle_deg"
                    ],
                    selected_angle,
                )
            ),

        "query_distance_m":
            float(
                query["distance_m"]
            ),
        "query_target_height_m":
            float(
                target_height_m
            ),
        "selected_distance_m":
            float(
                selected_distance
            ),
        "distance_selection_mode":
            str(
                distance_selection_mode
            ),
        "distance_error_m":
            float(
                abs(
                    query["distance_m"]
                    -
                    selected_distance
                )
            ),

        "query_elevation_deg":
            float(
                query["elevation_deg"]
            ),

        "selected_elevation_deg":
            float(
                selected_elevation
            ),

        "elevation_error_deg":
            float(
                abs(
                    query["elevation_deg"]
                    -
                    selected_elevation
                )
            ),

        "sprite_path":
            str(sprite_path),

        "exists":
            sprite_path.exists(),
        "anchor_x":
            float(record["anchor_x"]),

        "anchor_y":
            float(record["anchor_y"]),

        "source_sprite_width_px":
            int(record["sprite_width_px"]),

        "source_sprite_height_px":
            int(
                record[
                    "sprite_height_px"
                ]
            ),

        # =====================================================
        # Rich production metadata
        #
        # Propagated only. Rendering still uses anchor_x/y.
        # =====================================================

        "source_target_height_m":
            record.get(
                "target_height_m"
            ),

        "source_ground_anchor_x":
            record.get(
                "source_ground_anchor_x"
            ),

        "source_ground_anchor_y":
            record.get(
                "source_ground_anchor_y"
            ),

        "projected_ground_anchor_x_px":
            record.get(
                "projected_ground_anchor_x_px"
            ),

        "projected_ground_anchor_y_px":
            record.get(
                "projected_ground_anchor_y_px"
            ),

        "crop_x1_px":
            record.get(
                "crop_x1_px"
            ),

        "crop_y1_px":
            record.get(
                "crop_y1_px"
            ),

        "crop_x2_px":
            record.get(
                "crop_x2_px"
            ),

        "crop_y2_px":
            record.get(
                "crop_y2_px"
            ),

        "bbox_center_distance_m":
            record.get(
                "bbox_center_distance_m"
            ),

        "camera_forward_distance_m":
            record.get(
                "camera_forward_distance_m"
            ),

        "support_depth_m":
            record.get(
                "support_depth_m"
            ),

        "nearest_bbox_depth_m":
            record.get(
                "nearest_bbox_depth_m"
            ),

        "farthest_bbox_depth_m":
            record.get(
                "farthest_bbox_depth_m"
            ),

        "projected_bbox_center_x_px":
            record.get(
                "projected_bbox_center_x_px"
            ),

        "projected_bbox_bottom_y_px":
            record.get(
                "projected_bbox_bottom_y_px"
            ),

        "projected_bbox_width_px":
            record.get(
                "projected_bbox_width_px"
            ),

        "projected_bbox_height_px":
            record.get(
                "projected_bbox_height_px"
            ),

        "he_reference_projected_width_px":
            record.get(
                "he_reference_projected_width_px"
            ),

        "he_reference_projected_height_px":
            record.get(
                "he_reference_projected_height_px"
            ),

        "visible_width_px":
            record.get(
                "visible_width_px"
            ),

        "visible_height_px":
            record.get(
                "visible_height_px"
            ),

        "asset_id":
            record.get(
                "asset_id"
            ),

        "asset_class":
            record.get(
                "asset_class"
            ),

        "carla_blueprint":
            record.get(
                "carla_blueprint"
            ),

        "qa_pass":
            record.get(
                "qa_pass"
            ),

        "qa_flags":
            record.get(
                "qa_flags"
            ),

        "qa_warnings":
            record.get(
                "qa_warnings"
            ),
    }


def discover_available_sprite_angles(sprite_bank):
    root = Path(sprite_bank["root"])
    rgba_dir = sprite_bank.get("rgba_dir", "rgba")
    angle_format = sprite_bank.get("angle_format", "angle_{angle:03d}_rgba.png")

    sprite_dir = root / rgba_dir

    if not sprite_dir.exists():
        raise FileNotFoundError(f"Sprite RGBA directory not found: {sprite_dir}")

    available = []

    for angle in range(360):
        fname = angle_format.format(angle=angle)
        path = sprite_dir / fname
        if path.exists():
            available.append(angle)

    if not available:
        raise FileNotFoundError(f"No sprite files found in: {sprite_dir}")

    return available


def select_sprite(sprite_bank, relative_angle_deg, available_angles):
    root = Path(sprite_bank["root"])
    rgba_dir = sprite_bank.get("rgba_dir", "rgba")
    angle_format = sprite_bank.get("angle_format", "angle_{angle:03d}_rgba.png")

    selected_angle, angle_error = nearest_available_angle(
        relative_angle_deg,
        available_angles
    )

    sprite_path = root / rgba_dir / angle_format.format(angle=selected_angle)

    return {
        "relative_angle_deg": float(relative_angle_deg),
        "selected_angle": int(selected_angle),
        "angle_error_deg": float(angle_error),
        "sprite_path": str(sprite_path),
        "exists": sprite_path.exists()
    }


# ============================================================
# Sprite loading
# ============================================================

class SpriteCache:
    def __init__(self):
        self.cache = {}

    def load_rgba(self, path):
        path = str(path)

        if path in self.cache:
            return self.cache[path]

        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if rgba is None:
            raise RuntimeError(f"Could not read sprite: {path}")

        if len(rgba.shape) != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"Sprite must be BGRA PNG with alpha: {path}")

        # OpenCV gives BGRA. Convert to RGBA.
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)

        self.cache[path] = rgba
        return rgba


def resize_sprite_to_box(sprite_rgba, target_box_h):
    h, w = sprite_rgba.shape[:2]

    target_h = max(1, int(round(target_box_h)))
    scale = target_h / float(h)
    target_w = max(1, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    resized = cv2.resize(
        sprite_rgba,
        (target_w, target_h),
        interpolation=interp
    )

    return resized

def get_visible_alpha_bbox(
    sprite_rgba,
    alpha_threshold=10,
):
    """
    Compute the bounding box of the visible (alpha) part of a sprite.
    """

    alpha = sprite_rgba[:, :, 3]

    ys, xs = np.where(
        alpha > int(alpha_threshold)
    )

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "Sprite has no visible alpha pixels."
        )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": int(x2 - x1 + 1),
        "height": int(y2 - y1 + 1),
    }


def resize_view_matrix_sprite_to_box(
    sprite_rgba,
    target_box_w,
    target_box_h,
    anchor_x,
    anchor_y,
    alpha_threshold=10,
):
    """
    Resize a view-matrix sprite so that its VISIBLE alpha bounding box
    matches the HEPlacement target box width and height.

    Important separation of responsibilities:

        View matrix:
            selects appearance using
            (angle, distance, elevation)

        HEPlacement:
            determines final image-space geometry
            (cx, bottom_y, width, height)

    Therefore the selected sprite is allowed to use separate horizontal
    and vertical scaling. This prevents discrete distance/elevation sprite
    switches from causing artificial changes in rendered object width.

    Returns:
        resized_sprite_rgba
        resize_info
    """

    visible_bbox = get_visible_alpha_bbox(
        sprite_rgba=sprite_rgba,
        alpha_threshold=alpha_threshold,
    )

    visible_w = max(
        1,
        int(visible_bbox["width"]),
    )

    visible_h = max(
        1,
        int(visible_bbox["height"]),
    )

    target_w = max(
        1,
        float(target_box_w),
    )

    target_h = max(
        1,
        float(target_box_h),
    )

    # ------------------------------------------------------------
    # Independent geometry scales
    # ------------------------------------------------------------

    scale_x = (
        target_w
        / float(visible_w)
    )

    scale_y = (
        target_h
        / float(visible_h)
    )

    src_h, src_w = (
        sprite_rgba.shape[:2]
    )

    resized_w = max(
        1,
        int(
            round(
                src_w
                * scale_x
            )
        ),
    )

    resized_h = max(
        1,
        int(
            round(
                src_h
                * scale_y
            )
        ),
    )

    # If either axis enlarges the source, use linear interpolation.
    # Otherwise area interpolation is preferred for downsampling.
    if (
        scale_x < 1.0
        and
        scale_y < 1.0
    ):
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(
        sprite_rgba,
        (
            resized_w,
            resized_h,
        ),
        interpolation=interp,
    )

    resize_info = {
        "scale_x":
            float(scale_x),

        "scale_y":
            float(scale_y),

        # Retained for easier backwards inspection.
        # This is no longer the actual complete transform because
        # horizontal and vertical scaling are independent.
        "scale":
            float(scale_y),

        "source_width":
            int(src_w),

        "source_height":
            int(src_h),

        "visible_bbox":
            visible_bbox,

        "visible_width":
            int(visible_w),

        "visible_height":
            int(visible_h),

        "target_visible_width":
            float(target_w),

        "target_visible_height":
            float(target_h),

        "resized_width":
            int(resized_w),

        "resized_height":
            int(resized_h),

        "scaled_anchor_x":
            float(anchor_x)
            * float(scale_x),

        "scaled_anchor_y":
            float(anchor_y)
            * float(scale_y),
    }

    return (
        resized,
        resize_info,
    )
def warp_view_matrix_sprite_to_box_subpixel(
    sprite_rgba,
    frame_w,
    frame_h,
    target_cx,
    target_bottom_y,
    target_box_w,
    target_box_h,
    anchor_x,
    anchor_y,
    alpha_threshold=10,
):
    """
    Stable subpixel view-matrix renderer.

    Design invariant
    ----------------
    HEPlacement is the ONLY source of final image-space geometry:

        target_cx
        target_bottom_y
        target_box_w
        target_box_h

    The source sprite determines appearance only.

    Rendering uses:
      1. Exact continuous source -> target scale.
      2. INTER_AREA coarse prefilter for strong minification.
      3. A small residual affine transform for exact fractional scale
         and subpixel translation.

    IMPORTANT:
      - No post-raster alpha bbox is measured.
      - No raster measurement feeds back into scale.
      - No temporal smoothing.
      - No CARLA information.
      - No neighboring sprite blending.
      - No assumption that the actor is approaching or receding.

    Therefore this path can also be used later with reconstructed
    real-world dashcam scenarios.
    """

    # ============================================================
    # 1. Source geometry
    # ============================================================

    visible_bbox = get_visible_alpha_bbox(
        sprite_rgba=sprite_rgba,
        alpha_threshold=alpha_threshold,
    )

    visible_w = max(
        1,
        int(
            visible_bbox["width"]
        ),
    )

    visible_h = max(
        1,
        int(
            visible_bbox["height"]
        ),
    )

    target_w = max(
        1.0,
        float(target_box_w),
    )

    target_h = max(
        1.0,
        float(target_box_h),
    )

    source_h, source_w = (
        sprite_rgba.shape[:2]
    )

    # ============================================================
    # 2. Exact continuous geometry
    #
    # These are authoritative.
    # Nothing later is allowed to change them.
    # ============================================================

    exact_scale_x = (
        target_w
        /
        float(visible_w)
    )

    exact_scale_y = (
        target_h
        /
        float(visible_h)
    )

    exact_float_x1 = (
        float(target_cx)
        -
        float(anchor_x)
        *
        exact_scale_x
    )

    exact_float_y1 = (
        float(target_bottom_y)
        -
        float(anchor_y)
        *
        exact_scale_y
    )

    exact_float_x2 = (
        exact_float_x1
        +
        float(source_w)
        *
        exact_scale_x
    )

    exact_float_y2 = (
        exact_float_y1
        +
        float(source_h)
        *
        exact_scale_y
    )

    # ============================================================
    # 3. Coarse high-quality resize
    #
    # This is ONLY a numerical filtering stage.
    #
    # Importantly:
    #   coarse dimensions are derived directly from exact geometry.
    #
    # We never measure the resulting raster and modify geometry.
    # ============================================================

    coarse_w = max(
        1,
        int(
            round(
                float(source_w)
                *
                exact_scale_x
            )
        ),
    )

    coarse_h = max(
        1,
        int(
            round(
                float(source_h)
                *
                exact_scale_y
            )
        ),
    )

    coarse_scale_x = (
        float(coarse_w)
        /
        float(source_w)
    )

    coarse_scale_y = (
        float(coarse_h)
        /
        float(source_h)
    )

    # ------------------------------------------------------------
    # Premultiply before filtering.
    # ------------------------------------------------------------

    source_rgb = (
        sprite_rgba[:, :, :3]
        .astype(np.float32)
        /
        255.0
    )

    source_alpha = (
        sprite_rgba[:, :, 3]
        .astype(np.float32)
        /
        255.0
    )

    source_premul = (
        source_rgb
        *
        source_alpha[:, :, None]
    )

    if (
        exact_scale_x < 1.0
        and
        exact_scale_y < 1.0
    ):
        coarse_interp = (
            cv2.INTER_AREA
        )
    else:
        coarse_interp = (
            cv2.INTER_LINEAR
        )

    coarse_premul = cv2.resize(
        source_premul,
        (
            coarse_w,
            coarse_h,
        ),
        interpolation=coarse_interp,
    )

    coarse_alpha = cv2.resize(
        source_alpha,
        (
            coarse_w,
            coarse_h,
        ),
        interpolation=coarse_interp,
    )

    coarse_alpha = np.clip(
        coarse_alpha,
        0.0,
        1.0,
    )

    # ============================================================
    # 4. Residual scale
    #
    # THIS is the important difference from our previous version.
    #
    # residual = exact requested scale / actual coarse scale
    #
    # It is NOT calculated from a thresholded raster bbox.
    # ============================================================

    residual_scale_x = (
        exact_scale_x
        /
        coarse_scale_x
    )

    residual_scale_y = (
        exact_scale_y
        /
        coarse_scale_y
    )

    # Coarse-raster anchor derived analytically from the source.
    coarse_anchor_x = (
        float(anchor_x)
        *
        coarse_scale_x
    )

    coarse_anchor_y = (
        float(anchor_y)
        *
        coarse_scale_y
    )

    # ============================================================
    # 5. Exact floating-point placement
    #
    # This should algebraically equal exact_float_x1/y1 above.
    # ============================================================

    float_x1 = (
        float(target_cx)
        -
        coarse_anchor_x
        *
        residual_scale_x
    )

    float_y1 = (
        float(target_bottom_y)
        -
        coarse_anchor_y
        *
        residual_scale_y
    )

    float_x2 = (
        float_x1
        +
        float(coarse_w)
        *
        residual_scale_x
    )

    float_y2 = (
        float_y1
        +
        float(coarse_h)
        *
        residual_scale_y
    )

    # ============================================================
    # 6. Integer raster container
    #
    # Fractional location remains inside local_tx/local_ty.
    # ============================================================

    crop_x1 = int(
        math.floor(
            float_x1
        )
    )

    crop_y1 = int(
        math.floor(
            float_y1
        )
    )

    crop_x2 = int(
        math.ceil(
            float_x2
        )
    )

    crop_y2 = int(
        math.ceil(
            float_y2
        )
    )

    crop_w = max(
        1,
        crop_x2 - crop_x1,
    )

    crop_h = max(
        1,
        crop_y2 - crop_y1,
    )

    local_tx = (
        float_x1
        -
        float(crop_x1)
    )

    local_ty = (
        float_y1
        -
        float(crop_y1)
    )

    affine = np.array(
        [
            [
                float(
                    residual_scale_x
                ),
                0.0,
                float(
                    local_tx
                ),
            ],
            [
                0.0,
                float(
                    residual_scale_y
                ),
                float(
                    local_ty
                ),
            ],
        ],
        dtype=np.float32,
    )

    # ============================================================
    # 7. Small residual affine warp
    # ============================================================

    warped_premul = cv2.warpAffine(
        coarse_premul,
        affine,
        (
            crop_w,
            crop_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_alpha = cv2.warpAffine(
        coarse_alpha,
        affine,
        (
            crop_w,
            crop_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_alpha = np.clip(
        warped_alpha,
        0.0,
        1.0,
    )

    # ============================================================
    # 8. Convert premultiplied RGB back to straight RGBA
    # ============================================================

    warped_rgb = np.zeros_like(
        warped_premul,
        dtype=np.float32,
    )

    valid = (
        warped_alpha
        >
        1e-6
    )

    warped_rgb[valid] = (
        warped_premul[valid]
        /
        warped_alpha[
            valid,
            None,
        ]
    )

    warped_rgb = np.clip(
        warped_rgb,
        0.0,
        1.0,
    )

    warped_rgba = np.zeros(
        (
            crop_h,
            crop_w,
            4,
        ),
        dtype=np.uint8,
    )

    warped_rgba[:, :, :3] = (
        np.rint(
            warped_rgb
            *
            255.0
        )
        .astype(np.uint8)
    )

    warped_rgba[:, :, 3] = (
        np.rint(
            warped_alpha
            *
            255.0
        )
        .astype(np.uint8)
    )

    fully_outside_frame = (
        crop_x2 <= 0
        or
        crop_y2 <= 0
        or
        crop_x1 >= int(frame_w)
        or
        crop_y1 >= int(frame_h)
    )

    # ============================================================
    # 9. Numerical sanity checks
    #
    # These should be essentially zero.
    # ============================================================

    reconstructed_scale_x = (
        coarse_scale_x
        *
        residual_scale_x
    )

    reconstructed_scale_y = (
        coarse_scale_y
        *
        residual_scale_y
    )

    scale_error_x = (
        reconstructed_scale_x
        -
        exact_scale_x
    )

    scale_error_y = (
        reconstructed_scale_y
        -
        exact_scale_y
    )

    placement_error_x = (
        float_x1
        -
        exact_float_x1
    )

    placement_error_y = (
        float_y1
        -
        exact_float_y1
    )

    # ============================================================
    # 10. Metadata
    # ============================================================

    resize_info = {
        "mode":
            "subpixel_affine_area_no_feedback",

        "scale_x":
            float(
                exact_scale_x
            ),

        "scale_y":
            float(
                exact_scale_y
            ),

        # Compatibility.
        "scale":
            float(
                exact_scale_y
            ),

        "source_width":
            int(
                source_w
            ),

        "source_height":
            int(
                source_h
            ),

        "visible_bbox":
            visible_bbox,

        "visible_width":
            int(
                visible_w
            ),

        "visible_height":
            int(
                visible_h
            ),

        "target_visible_width":
            float(
                target_w
            ),

        "target_visible_height":
            float(
                target_h
            ),

        "coarse_width":
            int(
                coarse_w
            ),

        "coarse_height":
            int(
                coarse_h
            ),

        "coarse_scale_x":
            float(
                coarse_scale_x
            ),

        "coarse_scale_y":
            float(
                coarse_scale_y
            ),

        "residual_scale_x":
            float(
                residual_scale_x
            ),

        "residual_scale_y":
            float(
                residual_scale_y
            ),

        "reconstructed_scale_x":
            float(
                reconstructed_scale_x
            ),

        "reconstructed_scale_y":
            float(
                reconstructed_scale_y
            ),

        "scale_error_x":
            float(
                scale_error_x
            ),

        "scale_error_y":
            float(
                scale_error_y
            ),

        "placement_error_x":
            float(
                placement_error_x
            ),

        "placement_error_y":
            float(
                placement_error_y
            ),

        "resized_width":
            int(
                crop_w
            ),

        "resized_height":
            int(
                crop_h
            ),

        "scaled_anchor_x":
            float(target_cx)
            -
            float(crop_x1),

        "scaled_anchor_y":
            float(target_bottom_y)
            -
            float(crop_y1),

        "float_x1":
            float(
                float_x1
            ),

        "float_y1":
            float(
                float_y1
            ),

        "float_x2":
            float(
                float_x2
            ),

        "float_y2":
            float(
                float_y2
            ),

        "local_tx":
            float(
                local_tx
            ),

        "local_ty":
            float(
                local_ty
            ),

        "fully_outside_frame":
            bool(
                fully_outside_frame
            ),

        "paste": {
            "x1":
                int(
                    crop_x1
                ),

            "y1":
                int(
                    crop_y1
                ),

            "x2":
                int(
                    crop_x2
                ),

            "y2":
                int(
                    crop_y2
                ),

            "sprite_width":
                int(
                    crop_w
                ),

            "sprite_height":
                int(
                    crop_h
                ),
        },
    }

    return (
        warped_rgba,
        resize_info,
    )
# ============================================================
# Compositor
# ============================================================

def alpha_composite_rgb(frame_rgb, sprite_rgba, x1, y1, global_alpha=1.0):
    """
    frame_rgb: HxWx3 RGB uint8
    sprite_rgba: hxwx4 RGBA uint8
    x1,y1: top-left paste location

    Returns:
      composited RGB frame
      full-frame mask
    """

    frame_h, frame_w = frame_rgb.shape[:2]
    spr_h, spr_w = sprite_rgba.shape[:2]

    x2 = x1 + spr_w
    y2 = y1 + spr_h

    paste_x1 = max(0, x1)
    paste_y1 = max(0, y1)
    paste_x2 = min(frame_w, x2)
    paste_y2 = min(frame_h, y2)

    full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    if paste_x2 <= paste_x1 or paste_y2 <= paste_y1:
        return frame_rgb, full_mask

    sprite_x1 = paste_x1 - x1
    sprite_y1 = paste_y1 - y1
    sprite_x2 = sprite_x1 + (paste_x2 - paste_x1)
    sprite_y2 = sprite_y1 + (paste_y2 - paste_y1)

    sprite_crop = sprite_rgba[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    sprite_rgb = sprite_crop[:, :, :3].astype(np.float32)
    alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
    alpha = alpha * float(global_alpha)

    roi = frame_rgb[paste_y1:paste_y2, paste_x1:paste_x2].astype(np.float32)

    blended = sprite_rgb * alpha + roi * (1.0 - alpha)

    frame_rgb[paste_y1:paste_y2, paste_x1:paste_x2] = np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)

    full_mask[paste_y1:paste_y2, paste_x1:paste_x2] = np.clip(
        sprite_crop[:, :, 3].astype(np.float32) * float(global_alpha),
        0,
        255
    ).astype(np.uint8)

    return frame_rgb, full_mask

def get_frame_time(frame_idx, start_frame, fps):
    return (frame_idx - start_frame) / float(fps)

def draw_debug_box(frame_rgb, box, selected_angle=None):
    out = frame_rgb.copy()

    if not box.get("visible", False):
        return out

    x1 = int(round(box["x1"]))
    y1 = int(round(box["y1"]))
    x2 = int(round(box["x2"]))
    y2 = int(round(box["y2"]))

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    label = f"angle={selected_angle}" if selected_angle is not None else "adv"

    cv2.putText(
        out,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


# ============================================================
# Main temporal pipeline
# ============================================================

def run_scenario(scenario, overwrite=False):
    input_video_path = Path(scenario["input"]["video_path"])
    output_cfg = scenario["output"]

    ego_pose_records = None
    initial_pose_record = None

    coordinate_mode = scenario.get("coordinate_mode", {})
    runtime_transform = coordinate_mode.get("runtime_transform", "none")

    if runtime_transform == "ego_pose_jsonl":
        ego_pose_path = scenario.get("input", {}).get("ego_pose_path", None)

        if ego_pose_path is None:
            raise RuntimeError(
                "coordinate_mode.runtime_transform is ego_pose_jsonl, "
                "but input.ego_pose_path is missing."
            )

        ego_pose_records = load_ego_pose_jsonl(ego_pose_path)
        initial_pose_record = None

        print("[EgoPose] Loaded ego pose records:", len(ego_pose_records))
        print("[EgoPose] Path:", ego_pose_path)
        print("[EgoPose] Runtime transform enabled")

    output_dir = Path(output_cfg["output_dir"])

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_frames = bool(output_cfg.get("save_frames", False))
    save_masks = bool(output_cfg.get("save_masks", True))
    save_metadata = bool(output_cfg.get("save_metadata", True))

    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    debug_dir = output_dir / "debug"

    if save_frames:
        ensure_dir(frames_dir)

    if save_masks:
        ensure_dir(masks_dir)

    ensure_dir(debug_dir)

    output_video_name = output_cfg.get("output_video_name", "he_output.mp4")
    output_video_path = output_dir / output_video_name

    debug_video_path = output_dir / ("debug_" + output_video_name)

    cap = cv2.VideoCapture(str(input_video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # placement_adapter = build_placement_adapter_if_enabled(
    #     scenario=scenario,
    #     frame_w=frame_w,
    #     frame_h=frame_h
    # )
    lookup_path = scenario.get(
        "placement_lookup_npz",
        "assets/placement_lookup/heplacement_v2_lookup_full_0_100_z05_yaw1.npz",
    )

    placement_adapter = HEPlacementNPZLookupAdapter(
        lookup_path
    )

    # ============================================================
    # Optional geometry calibration
    #
    # Architecture:
    #
    #   HEPlacement
    #       ↓
    #   HEGeometryCalibrationV1
    #       ↓
    #   sprite selection / compositor
    #
    # The renderer itself remains unchanged.
    # ============================================================

    geometry_calibration_cfg = scenario.get(
        "geometry_calibration",
        {},
    )

    geometry_calibration = None

    if bool(
        geometry_calibration_cfg.get(
            "enabled",
            False,
        )
    ):

        calibration_type = (
            geometry_calibration_cfg.get(
                "type",
                "he_geometry_calibration_v1",
            )
        )

        if (
            calibration_type
            !=
            "he_geometry_calibration_v1"
        ):
            raise ValueError(
                "Unsupported geometry calibration type: "
                f"{calibration_type}"
            )

        calibration_path = (
            geometry_calibration_cfg.get(
                "npz_path"
            )
        )

        if not calibration_path:
            raise ValueError(
                "geometry_calibration.enabled=True "
                "but npz_path is missing."
            )

        geometry_calibration = (
            HEGeometryCalibrationV1(
                calibration_path
            )
        )

        print(
            "[HEGeometryCalibration] enabled"
        )

        print(
            "[HEGeometryCalibration] path:",
            calibration_path,
        )

    else:

        print(
            "[HEGeometryCalibration] disabled"
        )

    timeline = scenario["timeline"]

    start_frame = int(timeline["start_frame"])
    # In ego-pose-aware mode, "ego_initial" means ego pose at the
    # scenario start frame, not necessarily video frame 0.
    if runtime_transform == "ego_pose_jsonl":
        initial_pose_record = get_pose_record_for_frame(
            ego_pose_records=ego_pose_records,
            frame_idx=start_frame,
        )
        print("[EgoPose] Anchor frame for ego_initial:", start_frame)
    end_frame = int(timeline["end_frame"])
    scenario_fps = float(timeline.get("fps", input_fps if input_fps > 0 else 30.0))

    if end_frame < 0:
        end_frame = total_frames - 1

    end_frame = min(end_frame, total_frames - 1)

    camera = scenario["camera"]
    sprite_bank = scenario["sprite_bank"]
    adversaries = scenario["adversaries"]

    # Warn if scenario image size differs from actual video.
    if int(camera["image_width"]) != frame_w or int(camera["image_height"]) != frame_h:
        print("[WARN] Scenario camera image size differs from actual video.")
        print("       scenario:", camera["image_width"], "x", camera["image_height"])
        print("       video   :", frame_w, "x", frame_h)
        print("       Updating camera width/height to match video for this run.")
        camera["image_width"] = frame_w
        camera["image_height"] = frame_h
        camera["intrinsics"]["cx"] = frame_w / 2.0
        camera["intrinsics"]["cy"] = frame_h / 2.0

    output_fps = input_fps if input_fps > 0 else scenario_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (frame_w, frame_h))
    debug_writer = cv2.VideoWriter(str(debug_video_path), fourcc, output_fps, (frame_w, frame_h))

    sprite_bank_mode = (
        sprite_bank.get(
            "mode",
            "angle_only"
        )
    )

    render_transform_mode = (
        sprite_bank.get(
            "render_transform_mode",
            "integer_resize",
        )
    )

    if render_transform_mode not in (
        "integer_resize",
        "subpixel_affine",
    ):
        raise ValueError(
            "Unsupported render_transform_mode: "
            f"{render_transform_mode}"
        )

    print(
        "[HERender] transform mode:",
        render_transform_mode,
    )

    available_angles = None
    view_matrix = None

    if sprite_bank_mode == "view_matrix":

        view_matrix = (
            load_view_matrix_sprite_bank(
                sprite_bank
            )
        )

    elif sprite_bank_mode == "angle_only":

        available_angles = (
            discover_available_sprite_angles(
                sprite_bank
            )
        )

    else:

        raise ValueError(
            f"Unsupported sprite_bank mode: "
            f"{sprite_bank_mode}"
        )

    sprite_cache = SpriteCache()

    camera_yaw_deg = float(camera.get("yaw_deg", 0.0))

    metadata = {
        "scenario_id": scenario.get("scenario_id"),
        "input_video": str(input_video_path),
        "output_video": str(output_video_path),
        "debug_video": str(debug_video_path),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sprite_bank_mode":
            sprite_bank_mode,

        "render_transform_mode":
            render_transform_mode,

        "geometry_calibration": {
            "enabled":
                bool(
                    geometry_calibration
                    is not None
                ),

            "type":
                geometry_calibration_cfg.get(
                    "type",
                    None,
                ),

            "npz_path":
                geometry_calibration_cfg.get(
                    "npz_path",
                    None,
                ),
        },
        "available_sprite_count": (
            len(view_matrix["records"])
            if view_matrix is not None
            else len(available_angles)
        ),
        "frames": []
    }

    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        debug_rgb = frame_rgb.copy()

        frame_meta = {
            "frame_idx": frame_idx,
            "active": start_frame <= frame_idx <= end_frame,
            "adversaries": []
        }

        full_mask_accum = np.zeros((frame_h, frame_w), dtype=np.uint8)

        # Synthetic actors that are ready to render on this frame.
        # They are collected first, then composited far-to-near.
        render_candidates = []

        if start_frame <= frame_idx <= end_frame:

            t_sec = get_frame_time(
                frame_idx,
                start_frame,
                scenario_fps
            )
            
            for adv in adversaries:
                if not adv.get("enabled", True):
                    continue

                # ------------------------------------------------------------
                # Per-actor lifecycle
                #
                # ScenarioGenerator v2 exports exact actor existence as:
                #
                #     active_start_frame
                #     active_end_frame
                #
                # The actor must not be evaluated, placed, rendered, or added
                # to frame metadata outside this interval.
                #
                # Defaults preserve v1 behavior for old scenario files.
                # ------------------------------------------------------------

                active_start_frame = int(
                    adv.get(
                        "active_start_frame",
                        start_frame,
                    )
                )

                active_end_frame = int(
                    adv.get(
                        "active_end_frame",
                        end_frame,
                    )
                )

                if not (
                    active_start_frame
                    <= frame_idx
                    <= active_end_frame
                ):
                    continue

                denom = max(1, end_frame - start_frame)
                t_norm = (frame_idx - start_frame) / float(denom)
                scenario_local_frame_idx = frame_idx - start_frame

                raw_state = update_adversary_state(
                    adversary=adv,
                    t_sec=t_sec,
                    t_norm=t_norm,
                    frame_idx=scenario_local_frame_idx
                )

                state, transform_debug = transform_state_with_ego_pose_if_enabled(
                    state=raw_state,
                    scenario=scenario,
                    ego_pose_records=ego_pose_records,
                    initial_pose_record=initial_pose_record,
                    frame_idx=frame_idx,
                )

                # ------------------------------------------------------------
                # Visibility cull: do not render actor behind / too close to camera.
                # This is especially important in ego-pose-aware mode.
                # If z_m <= 0, the actor is behind the camera.
                # ------------------------------------------------------------
                min_render_depth_m = float(
                    scenario.get("visibility", {}).get("min_render_depth_m", 2.0)
                )

                if float(state.get("z_m", 0.0)) <= min_render_depth_m:
                    box = {
                        "visible": False,
                        "visible_prob": 0.0,
                        "reason": "behind_or_too_close_to_camera",
                        "z_m": float(state.get("z_m", 0.0)),
                        "min_render_depth_m": min_render_depth_m,
                        "source": "visibility_cull",
                    }

                    adv_meta = {
                        "id": adv["id"],
                        "state": state,
                        "raw_state": raw_state,
                        "transform_debug": transform_debug,
                        "coordinate_mode": scenario.get("coordinate_mode", {}),
                        "box": box,
                        "rendered": False,
                        "placement_source": "visibility_cull",
                        "debug_summary": {
                            "frame_idx": int(frame_idx),
                            "raw_x_m": float(raw_state.get("x_m", 0.0)),
                            "raw_z_m": float(raw_state.get("z_m", 0.0)),
                            "cam_x_m": float(state.get("x_m", 0.0)),
                            "cam_z_m": float(state.get("z_m", 0.0)),
                            "cam_yaw_deg": float(state.get("yaw_deg", 0.0)),
                            "cull_reason": "behind_or_too_close_to_camera",
                        },
                    }

                    frame_meta["adversaries"].append(adv_meta)
                    continue

                box = get_adversary_box(
                    state=state,
                    adversary=adv,
                    camera=camera,
                    image_width=frame_w,
                    image_height=frame_h,
                    placement_adapter=placement_adapter,
                    scenario=scenario
                )

                # ====================================================
                # HE Geometry Calibration V1
                #
                # Use the SAME continuous view-matrix coordinates used
                # by sprite selection:
                #
                #   distance_m
                #   viewpoint_angle_deg
                #
                # No selected sprite coordinates are involved.
                # ====================================================

                box_before_geometry_calibration = None
                geometry_calibration_query = None

                if (
                    geometry_calibration
                    is not None
                    and
                    box.get(
                        "visible",
                        False,
                    )
                ):

                    box_before_geometry_calibration = dict(
                        box
                    )

                    geometry_calibration_query = (
                        compute_view_matrix_coordinates(
                            state=state,

                            target_height_m=float(
                                sprite_bank.get(
                                    "target_height_m",
                                    0.75,
                                )
                            ),

                            vertical_mode=
                                sprite_bank.get(
                                    "vertical_mode",
                                    "state_y",
                                ),

                            camera_height_m=float(
                                sprite_bank.get(
                                    "camera_height_m",
                                    1.60,
                                )
                            ),
                        )
                    )

                    box = geometry_calibration.apply_box(
                        box=box,

                        distance_m=
                            geometry_calibration_query[
                                "distance_m"
                            ],

                        viewpoint_deg=
                            geometry_calibration_query[
                                "viewpoint_angle_deg"
                            ],
                    )

                adv_meta = {
                    "id": adv["id"],

                    # Final state after ego-pose transform.
                    # This is the state used by placement/compositing.
                    "state": state,

                    # Original generated state before ego-pose transform.
                    "raw_state": raw_state,

                    # Full transform debug information.
                    "transform_debug": transform_debug,

                    "coordinate_mode":
                        scenario.get(
                            "coordinate_mode",
                            {},
                        ),

                    # Final geometry used by the renderer.
                    "box":
                        box,

                    # Raw HEPlacement geometry before C(d, theta).
                    "box_before_geometry_calibration":
                        box_before_geometry_calibration,

                    # Continuous coordinates used to query C(d, theta).
                    "geometry_calibration_query":
                        geometry_calibration_query,

                    # Exact sampled correction and interpolation metadata.
                    "geometry_calibration":
                        box.get(
                            "geometry_calibration",
                            None,
                        ),

                    "rendered":
                        False,

                    # Preserve the true placement source even though
                    # the final box has passed through calibration.
                    "placement_source":
                        box.get(
                            "source_before_geometry_calibration",
                            box.get(
                                "source",
                                "geometric_projection",
                            ),
                        ),
                }

                if not box.get("visible", False):
                    frame_meta["adversaries"].append(adv_meta)
                    continue

                relative_angle = compute_sprite_angle_for_adversary(
                    state=state,
                    adversary=adv,
                    camera_yaw_deg=camera_yaw_deg,
                    frame_idx=frame_idx,
                    t_sec=t_sec,
                )

                if sprite_bank_mode == "view_matrix":

                    sprite_info = (
                        select_view_matrix_sprite(
                            state=state,
                            sprite_bank=sprite_bank,
                            view_matrix=view_matrix,
                        )
                    )

                else:

                    sprite_info = select_sprite(
                        sprite_bank=sprite_bank,
                        relative_angle_deg=relative_angle,
                        available_angles=available_angles,
                    )

                if not sprite_info["exists"]:
                    adv_meta["sprite"] = sprite_info
                    frame_meta["adversaries"].append(adv_meta)
                    continue

                sprite_rgba = sprite_cache.load_rgba(
                    sprite_info["sprite_path"]
                )

                if sprite_bank_mode == "view_matrix":

                    if (
                        render_transform_mode
                        ==
                        "subpixel_affine"
                    ):

                        sprite_resized, resize_info = (
                            warp_view_matrix_sprite_to_box_subpixel(
                                sprite_rgba=
                                    sprite_rgba,

                                frame_w=
                                    frame_w,

                                frame_h=
                                    frame_h,

                                target_cx=
                                    box["cx"],

                                target_bottom_y=
                                    box["bottom_y"],

                                target_box_w=
                                    box["box_width"],

                                target_box_h=
                                    box["box_height"],

                                anchor_x=
                                    sprite_info["anchor_x"],

                                anchor_y=
                                    sprite_info["anchor_y"],

                                alpha_threshold=
                                    10,
                            )
                        )

                        paste = (
                            resize_info[
                                "paste"
                            ]
                        )

                        paste_x1 = int(
                            paste["x1"]
                        )

                        paste_y1 = int(
                            paste["y1"]
                        )

                    else:

                        # ----------------------------------------------------
                        # Existing baseline integer-resize behavior.
                        # Keep unchanged for reproducibility.
                        # ----------------------------------------------------

                        sprite_resized, resize_info = (
                            resize_view_matrix_sprite_to_box(
                                sprite_rgba=
                                    sprite_rgba,

                                target_box_w=
                                    box["box_width"],

                                target_box_h=
                                    box["box_height"],

                                anchor_x=
                                    sprite_info["anchor_x"],

                                anchor_y=
                                    sprite_info["anchor_y"],
                            )
                        )

                        paste_x1 = int(
                            round(
                                box["cx"]
                                -
                                resize_info[
                                    "scaled_anchor_x"
                                ]
                            )
                        )

                        paste_y1 = int(
                            round(
                                box["bottom_y"]
                                -
                                resize_info[
                                    "scaled_anchor_y"
                                ]
                            )
                        )

                    spr_h, spr_w = (
                        sprite_resized.shape[:2]
                    )

                else:

                    # Legacy angle-only behavior.
                    sprite_resized = resize_sprite_to_box(
                        sprite_rgba=sprite_rgba,
                        target_box_h=box["box_height"]
                    )

                    spr_h, spr_w = sprite_resized.shape[:2]

                    paste_x1 = int(
                        round(
                            box["cx"]
                            - spr_w / 2.0
                        )
                    )

                    paste_y1 = int(
                        round(
                            box["bottom_y"]
                            - spr_h
                        )
                    )

                    resize_info = None

                alpha = float(
                    adv.get(
                        "rendering",
                        {}
                    ).get(
                        "alpha",
                        1.0
                    )
                )

                # ------------------------------------------------------------
                # Do NOT composite immediately.
                #
                # Collect all visible synthetic actors first so that they can
                # be depth sorted before rendering.
                # ------------------------------------------------------------

                adv_meta.update({
                    "relative_angle_deg": relative_angle,

                    "angle_mode": adv.get(
                        "rendering",
                        {}
                    ).get(
                        "angle_mode",
                        "viewpoint"
                    ),

                    "sprite": sprite_info,

                    "paste": {
                        "x1": paste_x1,
                        "y1": paste_y1,
                        "x2": paste_x1 + spr_w,
                        "y2": paste_y1 + spr_h,
                        "sprite_width": spr_w,
                        "sprite_height": spr_h,
                    },
                    "view_matrix_resize": resize_info,

                    "rendered": False,
                })

                adv_meta["debug_summary"] = {
                    "frame_idx": int(frame_idx),

                    "raw_x_m": float(
                        raw_state.get(
                            "x_m",
                            0.0
                        )
                    ),

                    "raw_z_m": float(
                        raw_state.get(
                            "z_m",
                            0.0
                        )
                    ),

                    "cam_x_m": float(
                        state.get(
                            "x_m",
                            0.0
                        )
                    ),

                    "cam_z_m": float(
                        state.get(
                            "z_m",
                            0.0
                        )
                    ),

                    "cam_yaw_deg": float(
                        state.get(
                            "yaw_deg",
                            0.0
                        )
                    ),

                    "box_center_x": float(
                        box.get(
                            "cx",
                            -1.0
                        )
                    ),

                    "box_bottom_y": float(
                        box.get(
                            "bottom_y",
                            -1.0
                        )
                    ),

                    "box_width": float(
                        box.get(
                            "box_width",
                            -1.0
                        )
                    ),

                    "box_height": float(
                        box.get(
                            "box_height",
                            -1.0
                        )
                    ),

                    "visible": bool(
                        box.get(
                            "visible",
                            False
                        )
                    ),

                    "placement_source": box.get(
                        "source",
                        "unknown"
                    ),
                }

                render_candidates.append({
                    "id": adv["id"],
                    "state": state,
                    "box": box,
                    "sprite_info": sprite_info,
                    "sprite_resized": sprite_resized,
                    "paste_x1": paste_x1,
                    "paste_y1": paste_y1,
                    "alpha": alpha,
                    "adv_meta": adv_meta,
                })

                # Metadata order remains scenario/actor order.
                frame_meta["adversaries"].append(
                    adv_meta
                )
        # ============================================================
        # Depth-sorted synthetic compositing
        #
        # Larger z = farther from camera.
        # Render far actors first and near actors last.
        # ============================================================

        render_candidates.sort(
            key=lambda item: float(
                item["state"].get(
                    "z_m",
                    0.0
                )
            ),
            reverse=True,
        )

        frame_meta["render_order"] = [
            {
                "id": item["id"],
                "z_m": float(
                    item["state"].get(
                        "z_m",
                        0.0
                    )
                ),
            }
            for item in render_candidates
        ]

        for render_index, item in enumerate(
            render_candidates
        ):
            box = item["box"]
            sprite_info = item["sprite_info"]
            sprite_resized = item["sprite_resized"]

            paste_x1 = item["paste_x1"]
            paste_y1 = item["paste_y1"]
            alpha = item["alpha"]

            adv_meta = item["adv_meta"]

            spr_h, spr_w = (
                sprite_resized.shape[:2]
            )

            frame_rgb, obj_mask = (
                alpha_composite_rgb(
                    frame_rgb=frame_rgb,
                    sprite_rgba=sprite_resized,
                    x1=paste_x1,
                    y1=paste_y1,
                    global_alpha=alpha,
                )
            )

            full_mask_accum = np.maximum(
                full_mask_accum,
                obj_mask,
            )

            debug_rgb = draw_debug_box(
                debug_rgb,
                box,
                selected_angle=
                    sprite_info["selected_angle"],
            )

            cv2.rectangle(
                debug_rgb,
                (
                    paste_x1,
                    paste_y1,
                ),
                (
                    paste_x1 + spr_w,
                    paste_y1 + spr_h,
                ),
                (0, 255, 0),
                1,
            )

            adv_meta["rendered"] = True

            adv_meta["render_order_index"] = int(
                render_index
            )

            adv_meta["render_depth_m"] = float(
                item["state"].get(
                    "z_m",
                    0.0
                )
            )
        metadata["frames"].append(frame_meta)

        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

        debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
        debug_writer.write(debug_bgr)

        if save_frames:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.png"), out_bgr)

        if save_masks:
            cv2.imwrite(str(masks_dir / f"frame_{frame_idx:06d}_mask.png"), full_mask_accum)

        frame_idx += 1

    cap.release()
    writer.release()
    debug_writer.release()

    if save_metadata:
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print("\n[HECompositor] Done.")
    print("[HECompositor] Scenario     :", scenario.get("scenario_id"))
    print("[HECompositor] Output video :", output_video_path)
    print("[HECompositor] Debug video  :", debug_video_path)
    print("[HECompositor] Metadata     :", output_dir / "metadata.json")
    print("[HECompositor] Frames       :", frame_idx)

def is_valid_predicted_box(box, image_width, image_height):
    if not box.get("visible", False):
        return False

    w = float(box.get("box_width", 0))
    h = float(box.get("box_height", 0))
    cx = float(box.get("cx", 0))
    bottom_y = float(box.get("bottom_y", 0))

    if w <= 1 or h <= 1:
        return False

    if cx < -image_width or cx > 2 * image_width:
        return False

    if bottom_y < -image_height or bottom_y > 2 * image_height:
        return False

    return True
# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default="configs/scenarios/oncoming_vehicle_001.json"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenario = load_json(args.scenario)

    run_scenario(
        scenario=scenario,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()