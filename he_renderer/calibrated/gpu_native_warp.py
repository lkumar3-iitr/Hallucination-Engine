"""Isolated resident native texture and inverse-warp candidate; same selector."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .fused_traversal import cp
from .gpu_close_pipeline import quantized_sampler
from .passby_renderer import aimed_camera, camera_rotation_map


class GPUNativeWarp:
    def __init__(self, native, budget_bytes=256*1024**2):
        if budget_bytes < 1:
            raise ValueError('Positive cache budget required')
        self.native = native
        self.budget = budget_bytes
        self.bytes = 0
        self.textures = OrderedDict()
        options = ('--fmad=false',)+(('-DQUANTIZED_LINEAR',) if quantized_sampler() else ())
        self.kernel = cp.RawKernel((Path(__file__).parent/'gpu_native_warp.cu').read_text(), 'warp', options=options)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def texture(self, path):
        # Banks are immutable while an engine is alive, matching the other caches.
        if path in self.textures:
            self.textures.move_to_end(path)
            return self.textures[path]
        sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
            raise ValueError(f'Invalid RGBA sprite: {path}')
        ys,xs = np.where(sprite[:,:,3]>10)
        if not len(xs):
            raise ValueError('Empty source sprite')
        crop = sprite[ys.min():ys.max()+1,xs.min():xs.max()+1]
        alpha = crop[:,:,3:4].astype(np.float32)/255
        source = cp.asarray(np.concatenate((crop[:,:,:3].astype(np.float32)*alpha,alpha),axis=2))
        while self.textures and self.bytes+source.nbytes>self.budget:
            _,old = self.textures.popitem(last=False)
            self.bytes -= old.nbytes
        if source.nbytes <= self.budget:
            self.textures[path] = source
            self.bytes += source.nbytes
        return source

    def render(self, background, actor, camera, fov, box_mode='hull', box_provider=None):
        return self._render(background, actor, camera, fov, box_mode, box_provider, False)

    def render_device(self, background, actor, camera, fov):
        """Keep rounded native output resident for downstream depth and blending."""
        return self._render(background, actor, camera, fov, 'hull', None, True)

    def _render(self, background, actor, camera, fov, box_mode, box_provider, device):
        if box_mode != 'hull' or box_provider is not None:
            return self.native.render(background,actor,camera,fov,box_mode,box_provider)
        height,width = background.shape[:2]
        virtual,_ = aimed_camera(actor,camera,self.selector.center,self.selector.extents)
        choice = self.selector.select(actor,virtual,width,height,fov)
        box = choice['predicted_box']
        if not np.all(np.isfinite(box)) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid full source-camera box')
        source = self.texture(choice['sprite_path'])
        matrix = cp.asarray(camera_rotation_map(virtual,camera,width,height,fov))
        image = cp.empty((height,width,3),cp.uint8)
        alpha = cp.empty((height,width),cp.float32)
        self.kernel(((height*width+127)//128,),(128,), (source,np.int32(source.shape[1]),np.int32(source.shape[0]),
            matrix,*map(np.float64,box),cp.asarray(background),image,alpha,np.int32(width),np.int32(height)))
        if device:
            visible = int(cp.count_nonzero(alpha > .04).item())
        else:
            image,alpha = cp.asnumpy(image),cp.asnumpy(alpha)
            visible = int(np.count_nonzero(alpha > .04))
        center = np.asarray(actor.get_matrix()) @ np.r_[self.selector.center,1.]
        distance = np.linalg.norm(center[:3]-np.array([camera.location.x,camera.location.y,camera.location.z]))
        meta = {k:v for k,v in choice.items() if k!='predicted_mask'}
        meta.update(box_mode=box_mode,virtual_box=box,center_distance_m=float(distance),
                    rendered=bool(visible),visible_pixels=visible)
        return image,alpha,meta
