"""Isolated bus candidate shared by offline evaluation and the runner adapter."""
from pathlib import Path
import json

import numpy as np

from .calibrated_close import CalibratedCloseBank, composite, coverage_weight
from .hull_rays import HullRays
from .passby_renderer import PassbyRenderer


class CalibratedBusRenderer:
    def __init__(self, native_bank, close_bank):
        root = Path(__file__).resolve().parent
        complete = json.loads((Path(close_bank)/"COMPLETE.json").read_text())
        self.native = PassbyRenderer(native_bank, root.parent/"cache/calibrated_v2")
        hull = HullRays(self.native.selector.faces)
        self.close = CalibratedCloseBank(Path(close_bank)/"view_matrix.csv",
                                        self.native.selector.center, 1.6, hull=hull)
        if complete["frames"] != len(self.close.rows):
            raise ValueError("Close-bank completion receipt does not match its rows")
        hull.refine(self.close, root.parent/"cache/calibrated_v2")

    def render(self, background, actor, camera, fov):
        h, w = background.shape[:2]
        result = self.close.render(actor, camera, w, h, fov)
        if result is None:
            image, alpha, meta = self.native.render(background, actor, camera, fov)
            meta.update(candidate_mode="native_fallback", close_weight=0.)
            return image, alpha, meta
        layer, meta = result
        weight = coverage_weight(self.close, meta["query_actor_local"])
        if weight < 1:
            native_image, native_alpha, _ = self.native.render(np.zeros_like(background), actor, camera, fov)
            layer = weight*layer + (1-weight)*np.dstack((native_image, native_alpha))
        query = -np.asarray(meta["query_actor_local"])
        meta.update(candidate_mode="calibrated_hull" if weight == 1 else "boundary_blend", close_weight=weight,
                    query=[float(np.degrees(np.arctan2(query[1], query[0])) % 360),
                           float(np.linalg.norm(query)),
                           float(np.degrees(np.arctan2(query[2], np.hypot(query[0], query[1]))))])
        return composite(background, layer), layer[:, :, 3], meta
