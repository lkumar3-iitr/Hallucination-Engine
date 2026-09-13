"""Exact integer coverage candidate; CPU projection and OpenCV fill retained."""
from pathlib import Path

import cv2
import numpy as np

from .fused_traversal import cp
from .selector import Selector, POPCOUNT_U8, project, relative_matrix


class GPUNativeOutline:
    def __init__(self, teacher):
        self.teacher = teacher
        code = (Path(__file__).parent/'gpu_native_outline.cu').read_text()
        self.collapsed = cp.RawKernel(code, 'collapsed')
        self.residual = cp.RawKernel(code, 'residual')
        self.intersections = cp.RawKernel(code, 'intersections')
        self.cached_faces = None
        self.cached_masks = None

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def selection_mask(self, actor, camera, width, height, fov):
        return self.predicted_mask(actor, camera, width, height, fov)

    def select(self, actor, camera, width, height, fov):
        if self.packed_masks.shape[1] % 8:
            return Selector.select(self, actor, camera, width, height, fov)
        predicted, box = self.selection_mask(actor,camera,width,height,fov)
        query = self.query(actor,camera)
        delta = np.abs((self.keys[:,0]-query[0]+180)%360-180)
        eligible = np.flatnonzero(delta<=35)
        if self.cached_masks is not self.packed_masks:
            self.device_masks = cp.asarray(np.ascontiguousarray(self.packed_masks))
            self.areas = POPCOUNT_U8[self.packed_masks].sum(axis=1)
            self.cached_masks = self.packed_masks
        resident = isinstance(predicted, cp.ndarray)
        target = cp.packbits(predicted.ravel() != 0) if resident else cp.asarray(np.packbits(predicted,axis=None))
        output = cp.empty(len(eligible),cp.int32)
        self.intersections((len(eligible),),(256,), (self.device_masks,target,cp.asarray(eligible,dtype=cp.int32),
            np.int32(self.packed_masks.shape[1]//8),output))
        if resident:
            # One host return supplies both ranking counts and the public CPU mask.
            packet = cp.asnumpy(cp.concatenate((output,target.astype(cp.int32))))
            counts = packet[:len(eligible)].astype(np.uint64)
            predicted = np.unpackbits(packet[len(eligible):].astype(np.uint8))[:self.size*self.size].reshape(self.size,self.size).astype(bool)
        else:
            counts = cp.asnumpy(output).astype(np.uint64)
        scores = counts/np.maximum(self.areas[eligible]+np.count_nonzero(predicted)-counts,1)
        local = int(np.argmax(scores-.002*(delta[eligible]/35)**2))
        selected = int(eligible[local])
        baseline = int(np.argmin(delta**2+(self.keys[:,2]-query[2])**2+(self.keys[:,1]-query[1])**2))
        return dict(index=selected,sprite_path=str(self.path(self.rows[selected])),key=self.keys[selected].tolist(),
                    query=query.tolist(),predicted_iou=float(scores[local]),baseline_index=baseline,
                    predicted_box=box,predicted_mask=predicted)

    def predicted_mask(self, actor, camera, width, height, fov):
        if self.cached_faces is not self.faces:
            self.vertices, inverse = np.unique(self.faces.reshape(-1,3), axis=0, return_inverse=True)
            self.indices = cp.asarray(inverse, dtype=cp.int32)
            self.cached_faces = self.faces
        focal = width/(2*np.tan(np.deg2rad(fov)/2))
        uv, depth = project(self.vertices, relative_matrix(actor, camera), focal, focal, width/2, height/2)
        if np.any(depth <= .01):
            raise ValueError('Hull intersects near plane; requires external clipping contract')
        low, high = uv.min(axis=0), uv.max(axis=0)
        if np.any(high-low < 1e-6):
            raise ValueError('Degenerate projected hull')
        vertices = cp.asarray(np.rint((uv-low)/(high-low)*(self.size-1)).astype(np.int32))
        count = len(self.faces)
        polygons = cp.empty((count,4,2),cp.int32)
        bounds = cp.empty((count,4),cp.int32)
        mask = cp.zeros((self.size,self.size),cp.int32)
        needed = cp.empty(count,cp.uint8)
        grid = ((count+127)//128,)
        self.collapsed(grid,(128,), (vertices,self.indices,mask,polygons,bounds,np.int32(count),np.int32(self.size)))
        self.residual(grid,(128,), (bounds,mask,needed,np.int32(count),np.int32(self.size)))
        remaining = cp.asnumpy(polygons[needed.astype(cp.bool_)])
        result = cp.asnumpy(mask).astype(np.uint8)
        records = remaining.reshape(-1,8).view(np.dtype((np.void,32))).ravel()
        _, indices = np.unique(records,return_index=True)
        for polygon in remaining[indices]:
            cv2.fillConvexPoly(result,polygon,1)
        return result.astype(bool), [*low.tolist(),*high.tolist()]
