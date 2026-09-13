"""Opt-in resident close pipeline. Exact CPU native selection remains available.

Own one instance per immutable calibrated engine; not thread-safe. Public arrays
are BGR and camera-forward metric depth, matching the existing HE core.
"""
from collections import OrderedDict
from copy import copy
import itertools
import math
from pathlib import Path

import numpy as np
import torch
import cv2

from .calibrated_close import bracket, camera_rays, coverage_weight
from .fused_traversal import cp
from .scene_depth import validate_depth
from .exact_selection_cache import ExactSelectionCache


def quantized_sampler():
    """Probe actual float RGBA remapping; do not infer it from a version string."""
    source = np.zeros((8, 8, 4), np.float32)
    source[:, :, :] = np.arange(8, dtype=np.float32)[None, :, None]
    x = np.full((4, 4), 2.123, np.float32)
    y = np.full_like(x, 3.456)
    value = cv2.remap(source, x, y, cv2.INTER_LINEAR)[0, 0, 0]
    if abs(value-x[0, 0]) < 1e-5:
        return False
    if abs(value-np.rint(x[0, 0]*32)/32) < 1e-5:
        return True
    raise RuntimeError('Unsupported OpenCV float interpolation; retain reference renderer')


class GPUClosePipeline:
    def __init__(self, engine, source_budget_bytes=512*1024**2, gpu_native=False, device_native=False):
        if source_budget_bytes < 1:
            raise ValueError('Positive source cache budget required')
        self.engine = engine
        self.device_native = device_native
        self.backend = 'he_calibrated_renderer_v2'
        self.native = copy(engine.native)
        teacher = engine.native.selector
        if gpu_native:
            from .gpu_native_outline import GPUNativeOutline
            teacher = GPUNativeOutline(teacher)
        self.native.selector = ExactSelectionCache(teacher)
        self.bank = engine.close
        self.hull = self.bank.hull
        if self.hull is None:
            raise ValueError('Calibrated hull required')
        torch.cuda.synchronize()
        self.volume = cp.from_dlpack(self.hull.volume)
        self.low, self.high = cp.asarray(self.hull.low, dtype=cp.float32), cp.asarray(self.hull.high, dtype=cp.float32)
        self.matrices = cp.asarray(np.asarray(self.bank.matrices))
        self.sources, self.rays = OrderedDict(), OrderedDict()
        self.source_bytes = 0
        self.budget = source_budget_bytes
        root = Path(__file__).resolve().parent
        self.march = cp.RawKernel((root/'fused_traversal.cu').read_text(), 'march', options=('--fmad=false',))
        code = (root/'gpu_close.cu').read_text()
        self.quantized = quantized_sampler()
        options = ('--fmad=false',) + (('-DQUANTIZED_LINEAR',) if self.quantized else ())
        self.sample = cp.RawKernel(code, 'source_sample', options=options)
        self.finish = cp.RawKernel(code, 'finish', options=('--fmad=false',))

    def source(self, index):
        if index in self.sources:
            self.sources.move_to_end(index)
            return self.sources[index]
        value = cp.asarray(self.bank.source(index))
        while self.sources and self.source_bytes+value.nbytes > self.budget:
            _, old = self.sources.popitem(last=False)
            self.source_bytes -= old.nbytes
        if value.nbytes <= self.budget:
            self.sources[index] = value
            self.source_bytes += value.nbytes
        return value

    def nodes(self, query):
        side = [v for v in self.bank.axes[1] if v*query[1] > 0]
        if not side:
            return None
        brackets = [bracket(v, q) for v, q in zip((self.bank.axes[0], side, self.bank.axes[2]), query)]
        if any(v is None for v in brackets):
            return None
        result = []
        for node in itertools.product(*brackets):
            weight = math.prod(v[1] for v in node)
            if weight < 1e-8:
                continue
            key = tuple(round(v[0], 6) for v in node)
            if key not in self.bank.index:
                return None
            result.append((self.bank.index[key], weight))
        return result

    def render(self, background, actor, camera, fov, scene_depth=None, include_occlusion_metadata=False):
        if background.dtype != np.uint8 or background.ndim != 3 or background.shape[2] != 3:
            raise ValueError('Expected uint8 HxWx3 BGR background')
        height, width = background.shape[:2]
        if not np.isfinite(fov) or not 0 < fov < 180:
            raise ValueError('FOV must be in (0, 180)')
        if scene_depth is not None:
            validate_depth(scene_depth, (height, width))
        target = np.asarray(actor.get_inverse_matrix()) @ np.asarray(camera.get_matrix())
        query = self.bank.center-target[:3, 3]
        nodes = self.nodes(query)
        weight = coverage_weight(self.bank, query) if nodes is not None else 0.
        native = None
        if nodes is None or weight < 1:
            # Preserve the reference's rounded native premultiplied boundary layer.
            native_render = self.native.render_device if self.device_native else self.native.render
            native = native_render(background if nodes is None else np.zeros_like(background), actor, camera, fov)
        if nodes is None and scene_depth is None:
            image, alpha, meta = native
            if self.device_native:
                image, alpha = cp.asnumpy(image), cp.asnumpy(alpha)
            return image, alpha, dict(meta, candidate_mode='native_fallback', close_weight=0.,
                                     performance_backend=self.backend)
        key = (width, height, float(fov))
        if key not in self.rays:
            self.rays[key] = cp.asarray(camera_rays(*key))
            if len(self.rays) > 16:
                self.rays.popitem(last=False)
        self.rays.move_to_end(key)
        transform = cp.asarray(target)
        directions = (self.rays[key] @ transform[:3, :3].T).astype(cp.float32)
        origin = cp.asarray(target[:3, 3], dtype=cp.float32)
        n = width*height
        points = cp.empty((n, 3), cp.float32)
        hits = cp.empty(n, cp.uint8)
        nz, ny, nx = self.hull.volume.shape[-3:]
        self.march(((n+127)//128,), (128,), (self.volume, directions, origin, self.low, self.high,
            np.float32(self.hull.spacing), np.int32(nx), np.int32(ny), np.int32(nz), np.int32(n), points, hits))
        if nodes is None:
            # Already composited native RGB: apply only depth, avoiding another alpha blend.
            layer = cp.concatenate((cp.asarray(native[0], dtype=cp.float32), cp.ones((height, width, 1), cp.float32)), axis=2)
            output_alpha = cp.asarray(native[1])
            meta = dict(native[2], candidate_mode='native_fallback', close_weight=0.)
        else:
            layer = cp.zeros((height, width, 4), cp.float32)
            for index, source_weight in nodes:
                source = self.source(index)
                row = self.bank.rows[index]
                calibration = [np.float64(row[k]) for k in ('camera_fx_px', 'camera_fy_px', 'camera_cx_px',
                               'camera_cy_px', 'crop_x1_px', 'crop_y1_px')]
                self.sample(((n+127)//128,), (128,), (points, hits, self.matrices[index], source,
                    np.int32(source.shape[1]), np.int32(source.shape[0]), *calibration,
                    np.float32(source_weight), layer, np.int32(n)))
            if weight < 1:
                if self.device_native:
                    other = cp.concatenate((native[0].astype(cp.float32), native[1][:, :, None]), axis=2)
                else:
                    other = cp.asarray(np.dstack((native[0], native[1])))
                layer = weight*layer+(1-weight)*other
            q = -query
            meta = dict(candidate_mode='calibrated_hull' if weight == 1 else 'boundary_blend', close_weight=weight,
                        query_actor_local=query.tolist(), query=[float(np.degrees(np.arctan2(q[1], q[0])) % 360),
                        float(np.linalg.norm(q)), float(np.degrees(np.arctan2(q[2], np.hypot(q[0], q[1]))))])
            meta['sources'] = [dict(index=index, weight=source_weight,
                node=[float(self.bank.rows[index][k]) for k in ('close_forward_m', 'close_right_m', 'close_target_up_m')])
                for index, source_weight in nodes]
        scene = cp.asarray(scene_depth, dtype=cp.float64) if scene_depth is not None else cp.empty(1, cp.float64)
        image = cp.empty((height, width, 3), cp.uint8)
        alpha = cp.empty((height, width), cp.float32)
        status = cp.empty((height, width), cp.uint8)
        self.finish(((n+127)//128,), (128,), (points, hits, transform, layer, cp.asarray(background), scene,
            np.int32(scene_depth is not None), image, alpha, status, np.int32(n)))
        if nodes is None:
            alpha *= output_alpha
        if scene_depth is not None and include_occlusion_metadata:
            visible = (output_alpha if nodes is None else layer[:, :, 3]) > 10/255
            counts = cp.asnumpy(cp.stack([cp.count_nonzero(visible),
                cp.count_nonzero(visible & ((status & 1) != 0)),
                cp.count_nonzero(visible & ((status & 2) != 0)),
                cp.count_nonzero(visible & (~cp.isfinite(scene) | (scene <= 0)))]))
            count, removed, unknown, invalid = map(int, counts)
            meta['scene_occlusion'] = dict(enabled=True, depth_convention='camera_forward_metres',
                depth_source='bank_visual_hull', tolerance_m=.05, input_visible_pixels=count,
                occluded_pixels=removed, occluded_fraction=removed/count if count else 0.,
                unknown_surface_pixels=unknown, invalid_scene_pixels=invalid)
        meta['performance_backend'] = self.backend
        return cp.asnumpy(image), cp.asnumpy(alpha), meta
