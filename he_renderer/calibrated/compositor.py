"""Opt-in bus candidate; other actors retain their current production backend."""
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from .calibrated_asset_renderer import CalibratedAssetRenderer
from .hull_rays import HullRays
from .scene_depth import occlude, surface_depth, validate_depth
from he_renderer.compositor import HESpriteRendererCompositor
from he_renderer.renderer import HESpriteRenderer
from he_multi_actor_compositor_v1 import (
    ActorRenderResult, MultiActorCompositeResult, actor_to_camera_state, make_actor_transform,
)


class HECalibratedCompositor:
    bank_path = None
    diagnostic_log = None
    bank_paths = {}
    gpu_close_enabled = True
    gpu_native_enabled = True

    def __init__(self, calibrated_manifest=None, pedestrian_gpu=False, **kwargs):
        import json
        manifest = Path(calibrated_manifest) if calibrated_manifest else Path(__file__).resolve().parents[1]/"manifests/calibrated_renderer_v2.json"
        config = json.loads(manifest.read_text())
        self.bank_paths = {row["blueprint"]: (manifest.parent/row["close_bank"]).resolve()
                           for row in config["assets"] if row.get("enabled", True)}
        for path in self.bank_paths.values():
            if not (path/"COMPLETE.json").is_file():
                raise FileNotFoundError(f"Missing calibrated asset: {path}")
        self.base = HESpriteRendererCompositor(**kwargs)
        self.bus_engines = {}
        self.calls = 0
        self.depth_hulls = {}
        self.gpu_engines = {}
        self.pedestrian_gpu = pedestrian_gpu
        self.pedestrian_depth = {}

    def render(self, base_rgb, camera_tf, active_actors, width, height, fov, scene_depth_m=None):
        if scene_depth_m is not None:
            validate_depth(scene_depth_m, (height, width))
        if base_rgb.shape[:2] != (height, width):
            raise ValueError("Frame/camera size mismatch")
        pairs = [(actor, actor_to_camera_state(actor, camera_tf)) for actor in active_actors]
        pairs.sort(key=lambda pair: pair[1].distance_euclidean_m, reverse=True)
        rgb, results = base_rgb.copy(), []
        root = Path(__file__).resolve().parent
        for order, (actor, state) in enumerate(pairs):
            bank = Path(state.he_view_matrix_csv).resolve().parent
            candidate_bank = self.bank_paths.get(state.carla_blueprint)
            if candidate_bank is None and state.carla_blueprint == "vehicle.mitsubishi.fusorosa":
                candidate_bank = self.bank_path
            if candidate_bank is None:
                if not state.carla_blueprint.startswith("walker.pedestrian."):
                    raise ValueError(f"No calibrated bank registered for {state.carla_blueprint}")
                if str(bank) not in self.base.engines:
                    table = bank/"selector_teacher_v1.npz"
                    self.base.engines[str(bank)] = HESpriteRenderer(
                        bank, root.parent/"cache/legacy",
                        selector_table=table if table.exists() else None, selector_similarity_threshold=.935)
                    if self.pedestrian_gpu:
                        from .gpu_pedestrian import accelerate_native
                        self.base.engines[str(bank)] = accelerate_native(self.base.engines[str(bank)])
                baseline = self.base.engines[str(bank)]
                original_render = baseline.render
                original_close = self.base.close_renderer._render_close_cartesian
                if scene_depth_m is not None:
                    def depth_render(background, actor_tf, camera, camera_fov):
                        image, alpha, metadata = original_render(background, actor_tf, camera, camera_fov)
                        if str(bank) not in self.depth_hulls:
                            self.depth_hulls[str(bank)] = HullRays(baseline.selector.faces)
                            if self.pedestrian_gpu:
                                from .gpu_pedestrian import PedestrianDepth
                                self.pedestrian_depth[str(bank)] = PedestrianDepth(self.depth_hulls[str(bank)])
                        if self.pedestrian_gpu:
                            depth = self.pedestrian_depth[str(bank)].surface_depth(actor_tf, camera, width, height, camera_fov)
                        else:
                            depth = surface_depth(self.depth_hulls[str(bank)], actor_tf, camera,
                                                  width, height, camera_fov)
                        image, alpha, info = occlude(image, background, alpha, depth, scene_depth_m)
                        metadata['scene_occlusion'] = info
                        return image, alpha, metadata

                    def depth_close(**kwargs):
                        kwargs['scene_depth_m'] = scene_depth_m
                        result = original_close(**kwargs)
                        if result is not None:
                            result[1].setdefault('scene_occlusion', {}).update(
                                depth_source='legacy_nearest_depth_approximation')
                        return result

                    baseline.render = depth_render
                    self.base.close_renderer._render_close_cartesian = depth_close
                try:
                    rendered = self.base.render(rgb, camera_tf, [actor], width, height, fov)
                finally:
                    baseline.render = original_render
                    self.base.close_renderer._render_close_cartesian = original_close
                rgb = rendered.rgb
                result = rendered.actor_results[0]
                result.render_order = order
                result.he_metadata["implementation_backend"] = result.he_metadata.get("renderer_backend")
                result.he_metadata["renderer_backend"] = "he_calibrated_renderer_v2"
                result.he_metadata["asset_backend"] = "legacy_pedestrian"
                if self.pedestrian_gpu:
                    result.he_metadata["performance_backend"] = "pedestrian_gpu_candidate_v1"
                results.append(result)
                continue
            if str(bank) not in self.bus_engines:
                self.bus_engines[str(bank)] = CalibratedAssetRenderer(bank, candidate_bank)
            engine = self.bus_engines[str(bank)]
            actor_tf = make_actor_transform(state.world_x, state.world_y, state.world_z, state.world_yaw_deg)
            background = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self.gpu_close_enabled:
                from .gpu_close_pipeline import GPUClosePipeline
                if str(bank) not in self.gpu_engines:
                    from .gpu_native_projection import GPUNativeProjection
                    from .gpu_native_warp import GPUNativeWarp
                    from .exact_selection_cache import ExactSelectionCache
                    pipeline = GPUClosePipeline(engine, gpu_native=True, device_native=True)
                    pipeline.native.selector = ExactSelectionCache(GPUNativeProjection(engine.native.selector, gpu_raster=True))
                    pipeline.native = GPUNativeWarp(pipeline.native)
                    self.gpu_engines[str(bank)] = pipeline
                image, alpha, meta = self.gpu_engines[str(bank)].render(
                    background, actor_tf, camera_tf, fov, scene_depth_m,
                    include_occlusion_metadata=True)
            else:
                image, alpha, meta = engine.render(background, actor_tf, camera_tf, fov)
            if scene_depth_m is not None and not self.gpu_close_enabled:
                depth = surface_depth(engine.close.hull, actor_tf, camera_tf, width, height, fov)
                image, alpha, info = occlude(image, background, alpha, depth, scene_depth_m)
                meta['scene_occlusion'] = info
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ys, xs = np.where(alpha > 10/255)
            bbox = None if not len(xs) else dict(x1=int(xs.min()), y1=int(ys.min()), x2=int(xs.max()), y2=int(ys.max()),
                width_px=int(xs.max()-xs.min()+1), height_px=int(ys.max()-ys.min()+1),
                center_x=float((xs.min()+xs.max())/2), bottom_y=float(ys.max()))
            rendered = bool(len(xs))
            query = meta["query"]
            key = meta.get("key", query)
            meta.update(renderer_backend=meta.get('performance_backend', 'he_calibrated_renderer_v2'), rendered=rendered,
                        visible_pixels=int(len(xs)),
                        reason=("" if rendered else "scene_depth_occluded"
                                if meta.get('scene_occlusion', {}).get('occluded_pixels', 0)
                                else "outside_viewport"), sprite_mode=meta["candidate_mode"],
                        selected_angle=key[0], selected_distance_m=key[1], selected_elevation_deg=key[2],
                        viewpoint_angle_deg=query[0], query_distance_m=query[1], query_elevation_deg=query[2],
                        rendered_alpha_bbox=bbox,
                        projection_mode=("pure_camera_rotation" if meta["candidate_mode"] == "native_fallback"
                                         else "calibrated_capture_hull_reprojection"),
                        anchor_mode="calibrated_rays", box={"visible": rendered, "depth_m": state.distance_forward_m,
                        "camera_right_m": state.rel_x_m, "camera_up_m": state.rel_y_m},
                        cartesian_close={"active": meta["close_weight"] > 0})
            shared = {f.name: getattr(state, f.name) for f in fields(ActorRenderResult) if hasattr(state, f.name)}
            results.append(ActorRenderResult(**shared, render_order=order, rendered=rendered, he_metadata=meta))
        if self.diagnostic_log is not None:
            import json
            self.diagnostic_log.write(json.dumps({"render_call": self.calls,
                "camera": {**{k: float(getattr(camera_tf.location, k)) for k in "xyz"},
                           **{k: float(getattr(camera_tf.rotation, k)) for k in ("yaw", "pitch", "roll")}},
                "actors": [{"actor_id": r.actor_id, "metadata": r.he_metadata} for r in results]})+"\n")
            self.diagnostic_log.flush()
        self.calls += 1
        return MultiActorCompositeResult(rgb=rgb, actor_results=results)
