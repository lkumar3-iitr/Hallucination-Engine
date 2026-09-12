"""Actual calibrated-bus rendering with controlled aligned depth fields."""
import json
from pathlib import Path
import carla
import cv2
import numpy as np

from .calibrated_bus_renderer import CalibratedBusRenderer
from .scene_depth import surface_depth, occlude


def main():
    root = Path(__file__).resolve().parent
    output = root/'artifacts/depth_smoke_v1'
    output.mkdir(exist_ok=False)
    engine = CalibratedBusRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/bus_aimed_close_v3')
    actor = carla.Transform(carla.Location(y=-4.2))
    camera = carla.Transform(carla.Location(x=-6, z=2.3), carla.Rotation(yaw=-60))
    background = np.full((300, 400, 3), 80, np.uint8)
    rgb, alpha, _ = engine.render(background, actor, camera, 100)
    depth = surface_depth(engine.close.hull, actor, camera, 400, 300, 100)
    scene = np.full((300, 400), 1000.)
    same, same_alpha, _ = occlude(rgb, background, alpha, depth, scene)
    np.testing.assert_array_equal(same, rgb)
    np.testing.assert_array_equal(same_alpha, alpha)
    scene[:, 170:230] = .5
    hidden, hidden_alpha, meta = occlude(rgb, background, alpha, depth, scene)
    assert meta['occluded_pixels'] > 0
    np.testing.assert_array_equal(hidden[:, :170], rgb[:, :170])
    cv2.imwrite(str(output/'comparison.png'), np.hstack((rgb, hidden)))
    np.savez_compressed(output/'evidence.npz', alpha=alpha, occluded_alpha=hidden_alpha,
                        actor_depth=depth, scene_depth=scene)
    (output/'summary.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
