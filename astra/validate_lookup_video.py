"""Continuous synthetic bus pass, exact versus guarded native selection."""
import json
import argparse
from pathlib import Path
from types import MethodType
import carla
import cv2
import numpy as np
from .calibrated_bus_renderer import CalibratedBusRenderer
from .native_lookup_candidate import NativeLookup
from .fused_traversal import intersect
from .calibrated_close import camera_rays


def road_reference(camera):
    """Ray-project a level world road, aligned with the actor's longitudinal axis."""
    matrix = np.asarray(camera.get_matrix())
    rays = camera_rays(400, 300, 100) @ matrix[:3, :3].T
    down = rays[:, :, 2] < -1e-6
    t = -matrix[2, 3]/np.where(down, rays[:, :, 2], -1)
    points = matrix[:3, 3]+rays*t[:, :, None]
    x, y = points[:, :, 0], points[:, :, 1]
    image = np.full((300, 400, 3), (210, 195, 175), np.uint8)
    image[down] = (90, 94, 90)
    road = down & (y > -6.3) & (y < 6.3)
    image[road] = (72, 72, 72)
    for edge in (-6.3, 6.3):
        image[down & (np.abs(y-edge)<.08)] = (230, 230, 230)
    for line in (-2.1, 2.1):
        image[road & (np.abs(y-line)<.07) & (np.mod(x, 6)<3)] = (245, 245, 245)
    # Faint transverse references make longitudinal perspective visible.
    image[road & (np.mod(x, 5)<.045)] = (105, 105, 105)
    return image


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--road-reference', action='store_true')
    args = parser.parse_args()
    output = root/('artifacts/native_lookup_road_v1' if args.road_reference else 'artifacts/native_lookup_video_v1')
    output.mkdir(exist_ok=False)
    bank = Path('D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3')
    engine = CalibratedBusRenderer(bank, root/'artifacts/bus_aimed_close_v3')
    teacher = engine.native.selector
    lookup = NativeLookup(teacher, bank/'selector_teacher_v1.npz')
    # Both sides use the same isolated fused traversal; only native selection varies.
    engine.close.hull._intersect_uncached = MethodType(intersect, engine.close.hull)
    actor = carla.Transform(carla.Location(y=-4.2))
    background = np.full((300, 400, 3), 80, np.uint8)
    writer = cv2.VideoWriter(str(output/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1200, 660))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    rows = []
    try:
        for frame, distance in enumerate(np.linspace(40, -10, 101)):
            strips = [[], []]
            for yaw in (0, -60, 60):
                camera = carla.Transform(carla.Location(x=float(-distance), z=2.3), carla.Rotation(yaw=yaw))
                if args.road_reference:
                    background = road_reference(camera)
                outputs = []
                for selector in (teacher, lookup):
                    engine.native.selector = selector
                    outputs.append(engine.render(background, actor, camera, 100))
                a, b = outputs
                ma, mb = a[1]>10/255, b[1]>10/255
                union = ma | mb
                rows.append(dict(frame=frame, distance=float(distance), yaw=yaw,
                    mask_iou=float(np.count_nonzero(ma & mb)/max(1, union.sum())) if union.any() else 1.,
                    max_rgb=int(np.abs(a[0].astype(int)-b[0].astype(int)).max()),
                    changed_pixels=int(np.any(a[0]!=b[0], axis=2).sum()),
                    mode=b[2]['candidate_mode'], selection=b[2].get('selection_mode', 'close_or_blend'),
                    reference_key=a[2].get('key'), lookup_key=b[2].get('key')))
                strips[0].append(a[0]); strips[1].append(b[0])
            panel = np.zeros((660, 1200, 3), np.uint8)
            panel[30:330] = np.hstack(strips[0])
            panel[360:660] = np.hstack(strips[1])
            cv2.putText(panel, f'Exact native selector | synthetic road reference | frame {frame}', (8, 22), 0, .6, (255,255,255), 1)
            cv2.putText(panel, 'Guarded lookup | same calibrated close path | not CARLA ground truth', (8, 352), 0, .6, (255,255,255), 1)
            writer.write(panel)
            if frame % 10 == 0 or frame in (54, 55, 56, 59):
                cv2.imwrite(str(output/f'frame_{frame:03d}.jpg'), panel)
                print(f'frame {frame}', flush=True)
    finally:
        engine.native.selector = teacher
        writer.release()
        (output/'rows.json').write_text(json.dumps(rows, indent=2))
    result = dict(frames=101, views=len(rows), min_mask_iou=min(r['mask_iou'] for r in rows),
        changed_views=sum(r['changed_pixels']>0 for r in rows),
        lookup_views=sum(r['selection']=='distilled' for r in rows),
        max_rgb=max(r['max_rgb'] for r in rows))
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
