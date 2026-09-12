"""Offline warm renderer profile; no CARLA server or model inference required."""
import argparse
import cProfile
import json
import time
from pathlib import Path

import carla
import numpy as np
import torch

from .calibrated_bus_renderer import CalibratedBusRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if root not in args.output.resolve().parents or args.repeats < 1:
        parser.error('Use a new ASTRA output directory and positive repeats')
    args.output.mkdir(parents=True, exist_ok=False)
    renderer = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    background = np.zeros((300, 400, 3), np.uint8)
    cases = [(distance, yaw) for distance in (30, 6, -6) for yaw in (0, -60, 60)]
    actor = carla.Transform(carla.Location(x=0, y=-4.2))
    results = []
    profile = cProfile.Profile()
    for distance, yaw in cases:
        camera = carla.Transform(carla.Location(x=-distance, z=2.3), carla.Rotation(yaw=yaw))
        renderer.render(background, actor, camera, 100)
        times = []
        for _ in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            profile.enable()
            rgb, alpha, meta = renderer.render(background, actor, camera, 100)
            profile.disable()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        name = f'd{distance}_yaw{yaw}'
        np.savez_compressed(args.output/f'{name}.npz', rgb=rgb, alpha=alpha)
        results.append(dict(case=name, mean_ms=float(np.mean(times)), mode=meta['candidate_mode']))
    profile.dump_stats(str(args.output/'profile.pstats'))
    (args.output/'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
