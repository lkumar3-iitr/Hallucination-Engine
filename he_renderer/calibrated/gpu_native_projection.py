"""Isolated double-precision GPU projection with a rounding-boundary fallback."""
import cv2
import numpy as np
from pathlib import Path

from .fused_traversal import cp
from .gpu_native_outline import GPUNativeOutline
from .selector import relative_matrix


class GPUNativeProjection(GPUNativeOutline):
    def __init__(self, teacher, fixed_download=False, gpu_raster=False, resident_scoring=False):
        super().__init__(teacher)
        if resident_scoring and not gpu_raster:
            raise ValueError('Resident scoring requires GPU rasterization')
        self.resident_scoring = resident_scoring
        self.fixed_download = fixed_download
        self.gpu_raster = gpu_raster
        if gpu_raster:
            self.fill = cp.RawKernel((Path(__file__).parent/'gpu_convex_fill.cu').read_text(), 'convex_fill')
        self.projection_queries = 0
        self.projection_fallbacks = 0

    def predicted_mask(self, actor, camera, width, height, fov):
        return self._project_mask(actor, camera, width, height, fov, False)

    def selection_mask(self, actor, camera, width, height, fov):
        return self._project_mask(actor, camera, width, height, fov, self.resident_scoring)

    def _project_mask(self, actor, camera, width, height, fov, device):
        self.projection_queries += 1
        if self.cached_faces is not self.faces:
            self.vertices, inverse = np.unique(self.faces.reshape(-1,3),axis=0,return_inverse=True)
            self.device_vertices = cp.asarray(self.vertices,dtype=cp.float64)
            self.indices = cp.asarray(inverse,dtype=cp.int32)
            self.cached_faces = self.faces
        focal = width/(2*np.tan(np.deg2rad(fov)/2))
        matrix = cp.asarray(relative_matrix(actor,camera))
        camera_points = self.device_vertices @ matrix[:3,:3].T + matrix[:3,3]
        depth = camera_points[:,0]
        uv = cp.stack((width/2+focal*camera_points[:,1]/depth,
                       height/2-focal*camera_points[:,2]/depth),axis=1)
        low, high = uv.min(axis=0), uv.max(axis=0)
        limits = cp.asnumpy(cp.concatenate((low,high,depth.min().reshape(1))))
        if not np.isfinite(limits).all() or abs(limits[4]-.01)<1e-9:
            self.projection_fallbacks += 1
            return super().predicted_mask(actor,camera,width,height,fov)
        if limits[4]<=.01:
            raise ValueError('Hull intersects near plane; requires external clipping contract')
        if np.any(limits[2:4]-limits[:2]<1e-6):
            raise ValueError('Degenerate projected hull')
        normalized = (uv-low)/(high-low)*(self.size-1)
        # A numerical guard, not a proven global error bound on GPU arithmetic.
        ambiguous = cp.any(cp.abs(normalized-cp.floor(normalized)-.5)<1e-9)
        if bool(ambiguous):
            self.projection_fallbacks += 1
            return super().predicted_mask(actor,camera,width,height,fov)
        vertices = cp.rint(normalized).astype(cp.int32)
        count = len(self.faces)
        polygons = cp.empty((count,4,2),cp.int32)
        bounds = cp.empty((count,4),cp.int32)
        mask = cp.zeros((self.size,self.size),cp.int32)
        needed = cp.empty(count,cp.uint8)
        grid = ((count+127)//128,)
        self.collapsed(grid,(128,), (vertices,self.indices,mask,polygons,bounds,np.int32(count),np.int32(self.size)))
        self.residual(grid,(128,), (bounds,mask,needed,np.int32(count),np.int32(self.size)))
        if self.gpu_raster:
            self.fill(grid,(128,), (polygons,needed,mask,np.int32(count),np.int32(self.size),np.int32(0)))
            return (mask if device else cp.asnumpy(mask).astype(bool)),limits[:4].tolist()
        if self.fixed_download:
            # Avoid dynamic GPU compaction; trade additional bytes for one host return.
            packet = cp.concatenate((polygons.ravel(), needed.astype(cp.int32), mask.ravel()))
            host = cp.asnumpy(packet)
            remaining = host[:count*8].reshape(count,4,2)[host[count*8:count*9] != 0]
            result = host[count*9:].reshape(self.size,self.size).astype(np.uint8)
        else:
            remaining = cp.asnumpy(polygons[needed.astype(cp.bool_)])
            result = cp.asnumpy(mask).astype(np.uint8)
        records = remaining.reshape(-1,8).view(np.dtype((np.void,32))).ravel()
        _, indices = np.unique(records,return_index=True)
        for polygon in remaining[indices]:
            cv2.fillConvexPoly(result,polygon,1)
        return result.astype(bool),limits[:4].tolist()
