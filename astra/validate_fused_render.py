"""Complete bus rendering regression sweep; synthetic aligned foreground depth."""
import json
import time
from pathlib import Path
from types import MethodType
import carla
import cv2
import numpy as np
import torch
from .calibrated_bus_renderer import CalibratedBusRenderer
from .fused_traversal import intersect
from .scene_depth import surface_depth, occlude


def main():
    root = Path(__file__).resolve().parent
    output = root/'artifacts/fused_render_sweep_v1'
    output.mkdir(exist_ok=False)
    engine = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    hull = engine.close.hull
    reference = hull._intersect_uncached
    candidate = MethodType(intersect, hull)
    actor = carla.Transform(carla.Location(y=-4.2))
    background = np.full((300, 400, 3), 80, np.uint8)
    scene = np.full((300, 400), 1000.)
    scene[:, 170:230] = .5
    rows = []
    try:
        for distance in range(18, -13, -1):
            for yaw in (0, -60, 60):
                camera = carla.Transform(carla.Location(x=-distance, z=2.3), carla.Rotation(yaw=yaw))
                outputs, times = [], []
                for method in (reference, candidate):
                    hull._intersect_uncached = method
                    hull._intersection_cache = None
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    rgb, alpha, meta = engine.render(background, actor, camera, 100)
                    depth = surface_depth(hull, actor, camera, 400, 300, 100)
                    hidden, hidden_alpha, info = occlude(rgb, background, alpha, depth, scene)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                    outputs.append((rgb, alpha, hidden, hidden_alpha, info))
                a, b = outputs
                row = dict(distance=distance, yaw=yaw, mode=meta['candidate_mode'],
                    reference_ms=times[0], fused_ms=times[1],
                    rgb_max=int(np.abs(a[0].astype(int)-b[0].astype(int)).max()),
                    rgb_changed_pixels=int(np.any(a[0]!=b[0], axis=2).sum()),
                    alpha_max=float(np.abs(a[1]-b[1]).max()),
                    mask_changes=int(np.count_nonzero((a[1]>10/255)!=(b[1]>10/255))),
                    occluded_rgb_max=int(np.abs(a[2].astype(int)-b[2].astype(int)).max()),
                    occluded_mask_changes=int(np.count_nonzero((a[3]>10/255)!=(b[3]>10/255))))
                rows.append(row)
                if distance in (12, 6, 0, -6):
                    cv2.imwrite(str(output/f'd{distance}_yaw{yaw}.jpg'),
                        np.vstack((np.hstack((a[0], b[0])), np.hstack((a[2], b[2])))))
            print(f'distance {distance} complete', flush=True)
    finally:
        hull._intersect_uncached = reference
        hull._intersection_cache = None
        (output/'rows.json').write_text(json.dumps(rows, indent=2))
    summary = dict(views=len(rows), max_rgb=max(r['rgb_max'] for r in rows),
        max_alpha=max(r['alpha_max'] for r in rows), mask_changes=sum(r['mask_changes'] for r in rows),
        occluded_mask_changes=sum(r['occluded_mask_changes'] for r in rows),
        max_occluded_rgb=max(r['occluded_rgb_max'] for r in rows))
    for mode in sorted({r['mode'] for r in rows}):
        selected = [r for r in rows if r['mode']==mode and r['distance']!=18]
        if selected:
            summary[mode] = {k:float(np.median([r[k] for r in selected])) for k in ('reference_ms','fused_ms')}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
