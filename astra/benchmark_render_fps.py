"""Renderer-core throughput including selection, sampling, composition and depth."""
import json
import argparse
import cProfile
import time
from pathlib import Path
from types import MethodType
import carla
import numpy as np
import torch
from .calibrated_bus_renderer import CalibratedBusRenderer
from .scene_depth import surface_depth, occlude
from .fused_traversal import intersect
from .native_lookup_candidate import NativeLookup


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookup', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    output = root/('artifacts/render_profile_ablation_v1' if args.ablation else
                  'artifacts/render_fps_lookup_v1' if args.lookup else 'artifacts/render_fps_v1')
    if args.output:
        output = args.output.resolve()
    if root not in output.parents or args.repeats < 1:
        parser.error('Use an ASTRA output and positive repeat count')
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    engine = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    initialization_s = time.perf_counter()-start
    hull = engine.close.hull
    original = hull._intersect_uncached
    teacher = engine.native.selector
    lookup = NativeLookup(teacher,
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3/selector_teacher_v1.npz') if args.lookup or args.ablation else None
    guarded_select = lookup._select_uncached if lookup is not None else None
    background = np.full((300, 400, 3), 80, np.uint8)
    depth_scene = np.full((300, 400), 1000.)
    depth_scene[:, 170:230] = .5
    results = []
    try:
        backends = (('fused_candidate', 'exact_fused_cached', 'lookup_fused_cached') if args.ablation else
                    ('fused_candidate', 'lookup_fused_cached') if args.lookup else ('current', 'fused_candidate'))
        for backend in backends:
            profile = cProfile.Profile() if args.ablation else None
            hull._intersect_uncached = original if backend == 'current' else MethodType(intersect, hull)
            engine.native.selector = lookup if backend in ('lookup_fused_cached', 'exact_fused_cached') else teacher
            if lookup is not None:
                lookup._select_uncached = teacher.select if backend == 'exact_fused_cached' else guarded_select
            for cameras in (1, 3):
                for actors in (1, 2, 3):
                    samples, modes = [], {}
                    for repeat in range(args.repeats+1):
                        for distance in (30, 15, 6, 0, -6, -12):
                            hull._intersection_cache = None
                            if lookup is not None:
                                lookup.selection_cache.clear()
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            if profile is not None and repeat:
                                profile.enable()
                            for yaw in (0, -60, 60)[:cameras]:
                                camera = carla.Transform(carla.Location(x=-distance, z=2.3),
                                                         carla.Rotation(yaw=yaw))
                                states = [carla.Transform(carla.Location(x=float(i*3), y=-4.2)) for i in range(actors)]
                                states.sort(key=lambda a: (a.location.x+distance)**2+4.2**2, reverse=True)
                                rgb = background.copy()
                                for actor in states:
                                    image, alpha, meta = engine.render(rgb, actor, camera, 100)
                                    depth = surface_depth(hull, actor, camera, 400, 300, 100)
                                    rgb, _, _ = occlude(image, rgb, alpha, depth, depth_scene)
                                    if repeat:
                                        mode = meta['candidate_mode']
                                        modes[mode] = modes.get(mode, 0)+1
                            torch.cuda.synchronize()
                            if profile is not None:
                                profile.disable()
                            elapsed = time.perf_counter()-start
                            if repeat:
                                samples.append(elapsed)
                    row = dict(backend=backend, cameras=cameras, actors=actors,
                        frames=len(samples), camera_sets_per_s=len(samples)/sum(samples),
                        images_per_s=cameras*len(samples)/sum(samples),
                        median_set_ms=float(np.median(samples)*1000),
                        p95_set_ms=float(np.percentile(samples, 95)*1000),
                        set_ms=[x*1000 for x in samples], actor_view_modes=modes)
                    results.append(row)
                    print(json.dumps(row), flush=True)
            if profile is not None:
                profile.dump_stats(str(output/f'{backend}.pstats'))
    finally:
        hull._intersect_uncached = original
        engine.native.selector = teacher
        hull._intersection_cache = None
        (output/'summary.json').write_text(json.dumps(dict(initialization_s=initialization_s,
            measured_repeats=args.repeats, profiled=args.ablation,
            resolution='400x300 FOV100 per image', asset='bus only; instances share bank/hull',
            depth='synthetic foreground strip; no HE-to-HE depth buffer',
            results=results), indent=2))


if __name__ == '__main__':
    main()
