"""Opt-in pedestrian native acceleration without changing its selection policy."""
from .gpu_native_projection import GPUNativeProjection
from .gpu_native_warp import GPUNativeWarp
from .exact_selection_cache import ExactSelectionCache
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .fused_traversal import cp
from .calibrated_close import camera_rays


class PedestrianSelector(GPUNativeProjection):
    def select(self, actor, camera, width, height, fov):
        selected = self.teacher.distilled_select(actor, camera, width, height, fov)
        if selected is not None:
            return selected
        selected = super().select(actor, camera, width, height, fov)
        selected.update(selection_mode='exact', distilled_neighbor_similarity=None)
        return selected


def accelerate_native(native):
    native.selector = ExactSelectionCache(PedestrianSelector(native.selector, gpu_raster=True))
    return GPUNativeWarp(native)


class PedestrianDepth:
    """Immutable hull and bounded resident camera rays; public depth remains CPU."""
    def __init__(self, hull):
        torch.cuda.synchronize()
        self.volume = cp.from_dlpack(hull.volume)
        self.low = cp.asarray(hull.low, dtype=cp.float32)
        self.high = cp.asarray(hull.high, dtype=cp.float32)
        self.spacing = hull.spacing
        self.rays = OrderedDict()
        self.kernel = cp.RawKernel((Path(__file__).parent/'fused_traversal.cu').read_text(),
                                  'march', options=('--fmad=false',))

    def surface_depth(self, actor, camera, width, height, fov):
        key = (width,height,fov)
        if key not in self.rays:
            if len(self.rays) >= 8:
                self.rays.popitem(last=False)
            self.rays[key] = cp.asarray(camera_rays(width,height,fov), dtype=cp.float64)
        self.rays.move_to_end(key)
        transform = np.asarray(actor.get_inverse_matrix()) @ np.asarray(camera.get_matrix())
        rotation = cp.asarray(transform[:3,:3])
        origin = cp.asarray(transform[:3,3])
        directions = (self.rays[key] @ rotation.T).astype(cp.float32).reshape(-1,3)
        points = cp.empty_like(directions)
        hits = cp.empty(len(directions), cp.uint8)
        nz,ny,nx = self.volume.shape[-3:]
        self.kernel(((len(directions)+127)//128,), (128,),
            (self.volume,directions,origin.astype(cp.float32),self.low,self.high,
             np.float32(self.spacing),np.int32(nx),np.int32(ny),np.int32(nz),np.int32(len(directions)),points,hits))
        forward = (points.astype(cp.float64)-origin) @ rotation[:,0]
        depth = cp.where((hits != 0) & (forward > 0), forward, cp.nan)
        return cp.asnumpy(depth).reshape(height,width)
