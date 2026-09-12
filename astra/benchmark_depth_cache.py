"""Compare paired render/depth with cache disabled versus enabled, warm GPU."""
import json
import time
from pathlib import Path
import carla
import numpy as np
import torch
from .calibrated_bus_renderer import CalibratedBusRenderer
from .scene_depth import surface_depth, occlude


def main():
    root = Path(__file__).resolve().parent
    output = root/'artifacts/depth_cache_benchmark_v1'
    output.mkdir(exist_ok=False)
    engine = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    actor = carla.Transform(carla.Location(y=-4.2))
    background = np.zeros((300, 400, 3), np.uint8)
    scene = np.full((300, 400), 1000.)
    scene[:, 170:230] = .5
    results = []
    for yaw in (0, -60, 60):
        camera = carla.Transform(carla.Location(x=-6, z=2.3), carla.Rotation(yaw=yaw))
        samples = {False: [], True: []}
        reference = None
        for iteration in range(6):
            for enabled in (False, True):
                # No cross-frame cache benefit: only the duplicate depth query is reused.
                engine.close.hull._intersection_cache = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                rgb, alpha, _ = engine.render(background, actor, camera, 100)
                if not enabled:
                    engine.close.hull._intersection_cache = None
                depth = surface_depth(engine.close.hull, actor, camera, 400, 300, 100)
                rgb, alpha, _ = occlude(rgb, background, alpha, depth, scene)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                if iteration:
                    samples[enabled].append(elapsed)
                if reference is None:
                    reference = (rgb.copy(), alpha.copy(), depth.copy())
                for a, b in zip(reference, (rgb, alpha, depth)):
                    np.testing.assert_array_equal(a, b)
        results.append(dict(yaw=yaw, disabled_median_ms=float(np.median(samples[False])),
                            enabled_median_ms=float(np.median(samples[True])),
                            exact_rgb_alpha_depth=True))
    (output/'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
