"""Opt-in recorder-compatible ASTRA compositor; no center-depth candidate rejection."""
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .passby_renderer import PassbyRenderer
else:
    from passby_renderer import PassbyRenderer
from he_multi_actor_compositor_v1 import (
    ActorRenderResult, HEMultiActorCompositorV1, MultiActorCompositeResult,
    actor_to_camera_state, make_actor_transform,
)


class AstraCompositor:
    def __init__(self, distance_selection_mode="linear", bottom_y_offset_px=0,
                 silhouette_scale=1, warp_scale_mode="independent", viewpoint_lateral_sign=1):
        if bottom_y_offset_px != 0 or silhouette_scale != 1 or viewpoint_lateral_sign != 1:
            raise ValueError("ASTRA uses calibrated geometry; manual offset/scale/sign overrides are unsupported")
        if warp_scale_mode != "independent":
            raise ValueError("ASTRA requires independent box fitting")
        self.engines = {}
        self.close_renderer = HEMultiActorCompositorV1(
            distance_selection_mode=distance_selection_mode,
            bottom_y_offset_px=bottom_y_offset_px,
            silhouette_scale=silhouette_scale,
            warp_scale_mode=warp_scale_mode,
            viewpoint_lateral_sign=viewpoint_lateral_sign,
        )

    def render(self, base_rgb, camera_tf, active_actors, width, height, fov, scene_depth_m=None):
        if scene_depth_m is not None:
            raise ValueError("ASTRA recorder adapter does not yet implement scene-depth occlusion")
        if base_rgb.shape[:2] != (height, width):
            raise ValueError("Frame dimensions do not match camera")
        states = [actor_to_camera_state(actor, camera_tf) for actor in active_actors]
        states.sort(key=lambda state: state.distance_euclidean_m, reverse=True)
        rgb, actor_results = base_rgb.copy(), []
        for order, state in enumerate(states):
            bank = Path(state.he_view_matrix_csv).resolve().parent
            if str(bank) not in self.engines:
                self.engines[str(bank)] = PassbyRenderer(bank, Path(__file__).parent / "artifacts/cache")
            engine = self.engines[str(bank)]
            actor_tf = make_actor_transform(state.world_x, state.world_y, state.world_z, state.world_yaw_deg)
            close = self.close_renderer._render_close_cartesian(
                base_rgb=rgb, state=state, actor_tf=actor_tf, camera_tf=camera_tf,
                view_matrix=engine.view_matrix, width=width, height=height, fov=fov,
                scene_depth_m=None, fallback_meta=None,
            )
            if close is not None:
                rgb, metadata = close
                rendered = bool(metadata.get("rendered"))
                metadata.update(
                    renderer_backend="astra_passby_v1_close_hybrid",
                    projection_mode="cartesian_close_calibrated",
                    box={"visible": rendered, "depth_m": state.distance_forward_m,
                         "camera_right_m": state.rel_x_m, "camera_up_m": state.rel_y_m},
                )
            else:
                image, alpha, metadata = engine.render(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), actor_tf, camera_tf, fov
                )
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ys, xs = np.where(alpha > 10/255)
                bbox = None if not len(xs) else {
                    "x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()), "y2": int(ys.max()),
                    "width_px": int(xs.max()-xs.min()+1), "height_px": int(ys.max()-ys.min()+1),
                    "center_x": float((xs.max()+xs.min())/2), "bottom_y": float(ys.max())}
                rendered = bool(len(xs))
                metadata.update(
                    renderer_backend="astra_passby_v1", reason="" if rendered else "outside_viewport",
                    sprite_mode="view_matrix", selected_angle=metadata["key"][0],
                    selected_distance_m=metadata["key"][1], selected_elevation_deg=metadata["key"][2],
                    viewpoint_angle_deg=metadata["query"][0], query_distance_m=metadata["query"][1],
                    query_elevation_deg=metadata["query"][2], rendered_alpha_bbox=bbox,
                    anchor_mode="astra_centered_camera_reprojection", projection_mode="pure_camera_rotation",
                    box={"visible": rendered, "depth_m": state.distance_forward_m,
                         "camera_right_m": state.rel_x_m, "camera_up_m": state.rel_y_m},
                    cartesian_close={"active": False, "reason": "outside_close_bank"})
            shared = {field.name: getattr(state, field.name) for field in fields(ActorRenderResult)
                      if hasattr(state, field.name)}
            actor_results.append(ActorRenderResult(**shared, render_order=order,
                                                   rendered=rendered, he_metadata=metadata))
        return MultiActorCompositeResult(rgb=rgb, actor_results=actor_results)
