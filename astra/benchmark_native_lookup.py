"""Explicit offline native lookup experiment."""
import json
import time
from pathlib import Path
import carla
import numpy as np
from .passby_renderer import PassbyRenderer, aimed_camera
from .native_lookup_candidate import NativeLookup
from .selector import packed_iou_scores


def main():
    root = Path(__file__).resolve().parent
    output = root/'artifacts/native_lookup_v1'
    output.mkdir(exist_ok=False)
    bank = Path('D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3')
    renderer = PassbyRenderer(bank, root/'artifacts/cache')
    lookup = NativeLookup(renderer.selector, bank/'selector_teacher_v1.npz')
    rows = []
    actor = carla.Transform()
    for distance in (12.5, 25., 40.):
        for bearing in np.arange(7.5, 360, 15):
            a = np.deg2rad(bearing)
            camera = carla.Transform(carla.Location(x=float(distance*np.cos(a)),
                y=float(distance*np.sin(a)), z=2.3))
            virtual, _ = aimed_camera(actor, camera, renderer.selector.center, renderer.selector.extents)
            start = time.perf_counter()
            expected = renderer.selector.select(actor, virtual, 400, 300, 100)
            reference_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            actual = lookup.select(actor, virtual, 400, 300, 100)
            candidate_ms = (time.perf_counter()-start)*1000
            score = packed_iou_scores(renderer.selector.packed_masks[[actual['index']]], expected['predicted_mask'])[0]
            np.testing.assert_array_equal(expected['predicted_box'], actual['predicted_box'])
            rows.append(dict(distance=distance, bearing=float(bearing), mode=actual['selection_mode'],
                same_sprite=expected['index']==actual['index'],
                iou_loss=float(expected['predicted_iou']-score), reference_ms=reference_ms,
                candidate_ms=candidate_ms))
    result = dict(poses=len(rows), lookup_poses=sum(r['mode']=='distilled' for r in rows),
        same_sprite=sum(r['same_sprite'] for r in rows), max_iou_loss=max(r['iou_loss'] for r in rows),
        reference_median_ms=float(np.median([r['reference_ms'] for r in rows])),
        candidate_median_ms=float(np.median([r['candidate_ms'] for r in rows])))
    (output/'rows.json').write_text(json.dumps(rows, indent=2))
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
