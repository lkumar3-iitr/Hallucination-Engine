"""Isolated reuse of the existing guarded lookup with exact current hull bounds."""
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from he_renderer.selector import Selector as ExistingSelector
from .selector import POPCOUNT_U8, project, relative_matrix


class NativeLookup:
    def __init__(self, teacher, table):
        self.teacher = teacher
        self.mask_areas = POPCOUNT_U8[teacher.packed_masks].sum(axis=1)
        self.vertices = np.unique(teacher.faces.reshape(-1, 3), axis=0)
        self.distilled = None
        self.selection_cache = OrderedDict()
        ExistingSelector.load_distilled(self, table, .935)
        with np.load(table) as data:
            features = data['features']
            self.minimum = features[:, 1:].min(axis=0)
            self.maximum = features[:, 1:].max(axis=0)

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def projected_box(self, actor, camera, width, height, fov):
        focal = width/(2*np.tan(np.deg2rad(fov)/2))
        uv, depth = project(self.vertices, relative_matrix(actor, camera),
                            focal, focal, width/2, height/2)
        if np.any(depth <= .01):
            raise ValueError('Hull crosses the virtual near plane')
        return [*uv.min(axis=0).tolist(), *uv.max(axis=0).tolist()]

    def select(self, actor, camera, width, height, fov):
        key = (tuple(np.asarray(actor.get_matrix()).ravel()),
               tuple(np.asarray(camera.get_matrix()).ravel()), width, height, fov)
        if key not in self.selection_cache:
            self.selection_cache[key] = self._select_uncached(actor, camera, width, height, fov)
            if len(self.selection_cache) > 16:
                self.selection_cache.popitem(last=False)
        self.selection_cache.move_to_end(key)
        return deepcopy(self.selection_cache[key])

    def _select_uncached(self, actor, camera, width, height, fov):
        query = self.query(actor, camera)
        if np.all(query[1:] >= self.minimum) and np.all(query[1:] <= self.maximum):
            choice = ExistingSelector.distilled_select(self, actor, camera, width, height, fov)
            if choice is not None:
                return choice
        choice = self.teacher.select(actor, camera, width, height, fov)
        choice['selection_mode'] = 'exact_fallback'
        return choice
