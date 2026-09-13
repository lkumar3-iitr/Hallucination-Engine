"""Bounded exact-query sharing; no pose rounding or approximate sprite selection."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np


class ExactSelectionCache:
    def __init__(self, teacher, capacity=16):
        if capacity < 1:
            raise ValueError('Positive capacity required')
        self.teacher, self.capacity = teacher, capacity
        self.cache = OrderedDict()

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def clear(self):
        self.cache.clear()

    def select(self, actor, camera, width, height, fov):
        # Asset arrays are immutable for this engine lifetime, as in the teacher.
        key = (id(self.teacher.faces), id(self.teacher.packed_masks), id(self.teacher.keys), self.teacher.size,
               tuple(np.asarray(actor.get_matrix()).ravel()),
               tuple(np.asarray(camera.get_matrix()).ravel()), width, height, fov)
        if key not in self.cache:
            self.cache[key] = deepcopy(self.teacher.select(actor, camera, width, height, fov))
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return deepcopy(self.cache[key])
