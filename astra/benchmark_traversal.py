"""Uncached traversal comparison against the frozen pre-batching implementation."""
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import carla
import argparse
from .calibrated_bus_renderer import CalibratedBusRenderer
from .calibrated_close import camera_rays


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--early', action='store_true')
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = root/('artifacts/traversal_fused_v1' if args.fused else
                  'artifacts/traversal_early_v1' if args.early else 'artifacts/traversal_batch_v1')
    if args.output:
        output = args.output.resolve()
        if root not in output.parents:
            parser.error('Output must stay inside ASTRA')
    output.mkdir(exist_ok=False)
    with zipfile.ZipFile(root/'frozen/pre_traversal_batch_20260912.zip') as archive:
        scope = {'__name__': 'frozen_hull'}
        exec(compile(archive.read('hull_rays.py'), '<frozen_hull>', 'exec'), scope)
    reference = scope['HullRays']._intersect_uncached
    engine = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    hull = engine.close.hull
    candidate = type(hull)._intersect_uncached
    if args.early:
        from .early_traversal import intersect
        candidate = intersect
    if args.fused:
        from .fused_traversal import intersect
        candidate = intersect
    results = []
    for distance in (6, 0, -6):
        for yaw in (0, -60, 60):
            rotation = np.asarray(carla.Transform(rotation=carla.Rotation(yaw=yaw)).get_matrix())[:3, :3]
            rays = camera_rays(400, 300, 100) @ rotation.T
            origin = np.array([-distance, 4.2, 2.3])
            samples = [[], []]
            expected = None
            for repeat in range(4):
                for index, method in enumerate((reference, candidate)):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = method(hull, origin, rays)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    if repeat:
                        samples[index].append(elapsed)
                    if expected is None:
                        expected = actual
                    if not args.fused:
                        for a, b in zip(expected, actual):
                            np.testing.assert_array_equal(a, b)
            results.append(dict(distance=distance, yaw=yaw,
                reference_ms=float(np.median(samples[0])), candidate_ms=float(np.median(samples[1])),
                exact=all(np.array_equal(a, b) for a, b in zip(expected, actual)),
                mask_disagreements=int(np.count_nonzero(expected[1] != actual[1])),
                maximum_hit_point_error_m=float(np.max(np.abs(expected[0][expected[1] & actual[1]]-
                    actual[0][expected[1] & actual[1]]), initial=0))))
    (output/'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
