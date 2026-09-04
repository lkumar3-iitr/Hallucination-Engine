"""Hallucination Engine compositor V2.

V2 keeps the V1 far-bank renderer and replaces only close Cartesian rendering.
Close views are treated as samples of a camera-relative light field: adjacent
forward, lateral, and yaw captures are projected independently and blended in
premultiplied RGBA. Asset colors are intentionally left unchanged so a bank
handoff remains visible during validation.
"""

from __future__ import annotations

import math

import numpy as np

from he_camera_renderer import (
    alpha_composite_rgb,
    apply_scene_depth_occlusion_to_sprite,
    warp_view_matrix_sprite_to_box_subpixel,
)
from he_multi_actor_compositor_v1 import HEMultiActorCompositorV1


class HEMultiActorCompositorV2(HEMultiActorCompositorV1):
    """V1-compatible compositor with continuous close-view reconstruction."""

    version = "v2_direct_cartesian_light_field"
    fallback_far_geometry_mode = "sprite_alpha_metric"
    centered_close_geometry_mode = "sprite_alpha_width_proxy_height"
    centered_close_projection_mode = "oriented_2p5d_support"
    center_depth_alpha_bbox_anchor = True
    close_inner_margin_cells = 4.0
    close_outer_margin_cells = 2.5
    close_inner_min_bbox_clearance_m = 0.15
    close_handoff_forward_m = 0.75
    close_handoff_right_m = 0.5
    camera_rotation_reprojection = True
    camera_rotation_reprojection_min_width_fraction = 0.20
    camera_rotation_reprojection_min_bearing_deg = 30.0

    @staticmethod
    def _smoothstep(value):
        value = min(1.0, max(0.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    def _close_handoff_weight(self, selection):
        """Return zero at a close-bank edge and one in its stable interior."""
        query = selection["query"]
        bank = selection["bank"]
        forward_min = min(float(value) for value in bank["forward_values"])
        forward_max = max(float(value) for value in bank["forward_values"])
        right_min = min(float(value) for value in bank["right_values"])
        right_max = max(float(value) for value in bank["right_values"])
        forward_edge_distance = min(
            float(query["forward_m"]) - forward_min,
            forward_max - float(query["forward_m"]),
        )
        right_edge_distance = min(
            float(query["right_m"]) - right_min,
            right_max - float(query["right_m"]),
        )
        forward_weight = self._smoothstep(
            forward_edge_distance / self.close_handoff_forward_m
        )
        right_weight = self._smoothstep(
            right_edge_distance / self.close_handoff_right_m
        )
        return min(forward_weight, right_weight)

    @staticmethod
    def _blend_box_with_fallback(box, fallback_bbox, close_weight):
        if not fallback_bbox:
            return box
        fallback = {
            "x1": float(fallback_bbox["x1"]),
            "x2": float(fallback_bbox["x2"]),
            "y1": float(fallback_bbox["y1"]),
            "y2": float(fallback_bbox["y2"]),
        }
        return {
            key: fallback[key] + float(close_weight) * (float(box[key]) - fallback[key])
            for key in ("x1", "x2", "y1", "y2")
        } | {"scale": float(box["scale"])}

    @staticmethod
    def _circular_yaw_bracket(values, query):
        ordered = sorted(float(value) % 360.0 for value in values)
        query = float(query) % 360.0
        extended = ordered + [ordered[0] + 360.0]
        adjusted = query if query >= ordered[0] else query + 360.0
        for lower, upper in zip(extended, extended[1:]):
            if lower <= adjusted <= upper:
                weight = 0.0 if upper == lower else (adjusted - lower) / (upper - lower)
                return lower % 360.0, upper % 360.0, weight
        return ordered[0], ordered[0], 0.0

    @staticmethod
    def _close_sampling_query(bank, query):
        """Map an edge query onto the nearest sampled camera ray.

        Close-bank images encode both actor viewpoint and off-axis pinhole
        perspective. When the runtime lateral position lies inside or outside
        the sampled strip, clamping lateral position while retaining forward
        depth changes that perspective sharply. Scale forward depth by the
        same factor as lateral position so the sampled camera bearing is
        retained whenever the bank's forward range permits it.
        """
        runtime_forward = float(query["forward_m"])
        runtime_right = float(query["right_m"])
        right_min = min(float(value) for value in bank["right_values"])
        right_max = max(float(value) for value in bank["right_values"])
        sample_right = min(right_max, max(right_min, runtime_right))
        sample_forward = runtime_forward
        ray_preserved = True

        if (
            sample_right != runtime_right
            and abs(runtime_right) > 1e-6
            and sample_right * runtime_right > 0.0
        ):
            sample_forward *= sample_right / runtime_right

        forward_min = min(float(value) for value in bank["forward_values"])
        forward_max = max(float(value) for value in bank["forward_values"])
        clamped_forward = min(forward_max, max(forward_min, sample_forward))
        if clamped_forward != sample_forward:
            ray_preserved = False
        sample_forward = clamped_forward

        return {
            "forward_m": float(sample_forward),
            "right_m": float(sample_right),
            "runtime_bearing_deg": math.degrees(math.atan2(
                runtime_right, runtime_forward
            )),
            "sample_bearing_deg": math.degrees(math.atan2(
                sample_right, sample_forward
            )),
            "ray_preserved": bool(ray_preserved),
        }

    def _continuous_rows(self, selection):
        bank = selection["bank"]
        query = selection["query"]
        sampling_query = self._close_sampling_query(bank, query)
        forward_lower, forward_upper, forward_weight = self._bracket(
            bank["forward_values"], float(sampling_query["forward_m"])
        )
        right_lower, right_upper, right_weight = self._bracket(
            bank["right_values"], float(sampling_query["right_m"])
        )
        query_bearing = float(sampling_query["runtime_bearing_deg"])

        weighted = []
        for forward in {forward_lower, forward_upper}:
            fw = 1.0 if forward_lower == forward_upper else (
                1.0 - forward_weight if forward == forward_lower else forward_weight
            )
            for right in {right_lower, right_upper}:
                rw = 1.0 if right_lower == right_upper else (
                    1.0 - right_weight if right == right_lower else right_weight
                )
                source_bearing = math.degrees(math.atan2(
                    float(right), float(forward)
                ))
                # Cartesian edge queries can be outside the sampled lateral
                # offsets. Selecting the same relative yaw there changes the
                # visible face because the sampled and runtime bearings differ.
                # Compensate yaw so source_bearing - source_yaw remains equal
                # to query_bearing - query_yaw, preserving actor-local view.
                source_yaw_query = (
                    float(query["relative_yaw_deg"])
                    + source_bearing
                    - query_bearing
                ) % 360.0
                yaw_lower, yaw_upper, yaw_weight = self._circular_yaw_bracket(
                    bank["yaw_values"], source_yaw_query
                )
                for yaw in {yaw_lower, yaw_upper}:
                    yw = 1.0 if yaw_lower == yaw_upper else (
                        1.0 - yaw_weight if yaw == yaw_lower else yaw_weight
                    )
                    key = (
                        round(float(forward), 6),
                        round(float(right), 6),
                        round(float(yaw) % 360.0, 6),
                    )
                    row = bank["index"].get(key)
                    weight = fw * rw * yw
                    if row is not None and weight > 0.0:
                        weighted.append((weight, row))

        total = sum(weight for weight, unused in weighted)
        if total <= 0.0:
            return []
        return [(weight / total, row) for weight, row in weighted]

    @staticmethod
    def _runtime_box(row, query, width, height, runtime_fx):
        selected_forward = max(float(row["close_forward_m"]), 1e-6)
        query_forward = max(float(query["forward_m"]), 1e-6)
        capture_fx = float(row["camera_fx_px"])
        capture_fy = float(row["camera_fy_px"])
        scale = runtime_fx / capture_fx * selected_forward / query_forward

        query_center_x = float(width) / 2.0 + (
            runtime_fx * float(query["right_m"]) / query_forward
        )
        query_center_y = float(height) / 2.0 - (
            runtime_fx * float(query["up_m"]) / query_forward
        )
        source_center_x = float(row["camera_cx_px"]) + (
            capture_fx * float(row["close_right_m"]) / selected_forward
        )
        source_center_y = float(row["camera_cy_px"]) - (
            capture_fy * float(row["close_target_up_m"]) / selected_forward
        )
        return {
            "x1": query_center_x + (float(row["visible_x1_full_px"]) - source_center_x) * scale,
            "x2": query_center_x + (float(row["visible_x2_full_px"]) - source_center_x) * scale,
            "y1": query_center_y + (float(row["visible_y1_full_px"]) - source_center_y) * scale,
            "y2": query_center_y + (float(row["visible_y2_full_px"]) - source_center_y) * scale,
            "scale": scale,
        }

    def _render_close_cartesian(
        self,
        base_rgb,
        state,
        actor_tf,
        camera_tf,
        view_matrix,
        width,
        height,
        fov,
        scene_depth_m=None,
        fallback_meta=None,
    ):
        selection = self._select_close_bank(
            state, actor_tf, camera_tf, view_matrix
        )
        if selection is None:
            return None

        weighted_rows = self._continuous_rows(selection)
        if not weighted_rows:
            return None

        query = selection["query"]
        sampling_query = self._close_sampling_query(
            selection["bank"], query
        )
        if not sampling_query["ray_preserved"]:
            return None
        actual_right_min = min(
            float(value) for value in selection["bank"]["right_values"]
        )
        actual_right_max = max(
            float(value) for value in selection["bank"]["right_values"]
        )
        if not actual_right_min <= float(query["right_m"]) <= actual_right_max:
            return None

        runtime_fx = float(width) / (
            2.0 * math.tan(math.radians(float(fov)) / 2.0)
        )
        fallback_bbox = (
            (fallback_meta or {}).get("rendered_alpha_bbox")
            if (fallback_meta or {}).get("rendered", False)
            else None
        )
        handoff_weight = self._close_handoff_weight(selection)
        premultiplied = np.zeros((int(height), int(width), 3), dtype=np.float32)
        alpha_sum = np.zeros((int(height), int(width)), dtype=np.float32)
        boxes = []
        source_rows = []

        for weight, row in weighted_rows:
            sprite_path = self._close_row_path(selection["bank"], row)
            if not sprite_path.exists():
                continue
            sprite = self.sprite_cache.load_rgba(sprite_path)
            alpha_y, alpha_x = np.where(sprite[:, :, 3] > 10)
            if len(alpha_x) == 0:
                continue
            box = self._runtime_box(row, query, width, height, runtime_fx)
            box = self._blend_box_with_fallback(
                box, fallback_bbox, handoff_weight
            )
            warped, resize = warp_view_matrix_sprite_to_box_subpixel(
                sprite_rgba=sprite,
                frame_w=int(width),
                frame_h=int(height),
                target_cx=(box["x1"] + box["x2"]) / 2.0,
                target_bottom_y=box["y2"],
                target_box_w=max(1.0, box["x2"] - box["x1"] + 1.0),
                target_box_h=max(1.0, box["y2"] - box["y1"] + 1.0),
                anchor_x=(float(alpha_x.min()) + float(alpha_x.max())) / 2.0,
                anchor_y=float(alpha_y.max()),
                alpha_threshold=10,
            )
            if resize.get("fully_outside_frame", False):
                continue
            paste = resize["paste"]
            raw_x1, raw_y1 = int(paste["x1"]), int(paste["y1"])
            raw_x2 = raw_x1 + warped.shape[1]
            raw_y2 = raw_y1 + warped.shape[0]
            x1, y1 = max(0, raw_x1), max(0, raw_y1)
            x2, y2 = min(int(width), raw_x2), min(int(height), raw_y2)
            if x1 >= x2 or y1 >= y2:
                continue
            source_x1, source_y1 = x1 - raw_x1, y1 - raw_y1
            source_x2 = source_x1 + (x2 - x1)
            source_y2 = source_y1 + (y2 - y1)
            clipped = warped[source_y1:source_y2, source_x1:source_x2]
            alpha = clipped[:, :, 3].astype(np.float32) / 255.0
            weighted_alpha = float(weight) * alpha
            premultiplied[y1:y2, x1:x2] += (
                clipped[:, :, :3].astype(np.float32) * weighted_alpha[:, :, None]
            )
            alpha_sum[y1:y2, x1:x2] += weighted_alpha
            boxes.append((float(weight), box))
            source_rows.append((float(weight), row, str(sprite_path)))

        if not source_rows or not np.any(alpha_sum > 0.0):
            return None

        blended = np.zeros((int(height), int(width), 4), dtype=np.uint8)
        visible = alpha_sum > 1e-6
        blended[:, :, 3] = np.clip(alpha_sum * 255.0, 0.0, 255.0).astype(np.uint8)
        blended[visible, :3] = np.clip(
            premultiplied[visible] / alpha_sum[visible, None], 0.0, 255.0
        ).astype(np.uint8)

        dominant_weight, dominant_row, dominant_path = max(
            source_rows, key=lambda item: item[0]
        )
        nearest_depth = max(
            0.05,
            float(query["forward_m"])
            + float(dominant_row["nearest_bbox_depth_m"])
            - float(dominant_row["actor_depth_m"]),
        )
        blended, occlusion = apply_scene_depth_occlusion_to_sprite(
            sprite_rgba=blended,
            scene_depth_m=scene_depth_m,
            paste_x1=0,
            paste_y1=0,
            actor_nearest_depth_m=nearest_depth,
        )
        output, mask = alpha_composite_rgb(
            frame_rgb=base_rgb.copy(),
            sprite_rgba=blended,
            x1=0,
            y1=0,
            global_alpha=1.0,
        )

        total_box_weight = sum(weight for weight, unused in boxes)
        target = {
            name: sum(weight * box[name] for weight, box in boxes) / total_box_weight
            for name in ("x1", "x2", "y1", "y2")
        }
        requested_view = (
            math.degrees(math.atan2(float(query["right_m"]), float(query["forward_m"])))
            - float(query["relative_yaw_deg"])
            + 180.0
        ) % 360.0
        selected_view = (
            math.degrees(math.atan2(
                float(dominant_row["close_right_m"]),
                float(dominant_row["close_forward_m"]),
            ))
            - float(dominant_row["close_relative_yaw_deg"])
            + 180.0
        ) % 360.0

        return output, {
            "rendered": bool(np.any(mask > 0)),
            "reason": "",
            "renderer_version": self.version,
            "sprite_mode": "cartesian_close_continuous",
            "selected_angle": float(selected_view),
            "viewpoint_angle_deg": float(requested_view),
            "selected_distance_m": float(dominant_row["bbox_center_distance_m"]),
            "selected_elevation_deg": float(dominant_row["elevation_deg"]),
            "sprite_path": dominant_path,
            "sprite_width": int(width),
            "sprite_height": int(height),
            "paste_x1": 0,
            "paste_y1": 0,
            "anchor_mode": "v2_cartesian_light_field",
            "target_box_width_px": float(target["x2"] - target["x1"] + 1.0),
            "target_box_height_px": float(target["y2"] - target["y1"] + 1.0),
            "warp_scale_mode": "v2_premultiplied_trilinear",
            "scene_occlusion": occlusion,
            "cartesian_close": {
                "active": True,
                "side": selection["side"],
                "query": query,
                "dominant_weight": dominant_weight,
                "sample_count": len(source_rows),
                "handoff_weight_close": float(handoff_weight),
                "handoff_reference": (
                    "view_matrix_rendered_alpha_bbox"
                    if fallback_bbox is not None
                    else "none"
                ),
                "source_weights": [
                    {
                        "weight": weight,
                        "forward_m": float(row["close_forward_m"]),
                        "right_m": float(row["close_right_m"]),
                        "relative_yaw_deg": float(row["close_relative_yaw_deg"]),
                        "viewpoint_angle_deg": (
                            math.degrees(math.atan2(
                                float(row["close_right_m"]),
                                float(row["close_forward_m"]),
                            ))
                            - float(row["close_relative_yaw_deg"])
                            + 180.0
                        ) % 360.0,
                    }
                    for weight, row, unused in source_rows
                ],
                "sampling_query": self._close_sampling_query(
                    selection["bank"], query
                ),
                "runtime_projection": {"target_visible_box": target},
            },
        }
