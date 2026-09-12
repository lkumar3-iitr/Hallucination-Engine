"""Compare optimized native selection to the frozen polygon-loop baseline."""
import json
import argparse
import zipfile
from pathlib import Path
import numpy as np
import carla
from .passby_renderer import PassbyRenderer, aimed_camera


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root/'artifacts/native_runs_validation_v1')
    output = parser.parse_args().output.resolve()
    if root not in output.parents:
        parser.error('Output must stay inside ASTRA')
    output.mkdir(exist_ok=False)
    with zipfile.ZipFile(root/'frozen/pre_native_points_20260912.zip') as archive:
        scope = {'__name__': 'frozen_selector'}
        exec(compile(archive.read('selector.py'), '<frozen_selector>', 'exec'), scope)
    baseline = scope['Selector'].predicted_mask
    renderer = PassbyRenderer(
        'D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3',
        root/'artifacts/cache')
    actor = carla.Transform()
    count = 0
    for distance in (10, 20, 40):
        for bearing in range(0, 360, 15):
            angle = np.deg2rad(bearing)
            camera = carla.Transform(carla.Location(x=float(distance*np.cos(angle)),
                y=float(distance*np.sin(angle)), z=2.3))
            virtual, _ = aimed_camera(actor, camera, renderer.selector.center)
            a, box_a = baseline(renderer.selector, actor, virtual, 400, 300, 100)
            b, box_b = renderer.selector.predicted_mask(actor, virtual, 400, 300, 100)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(box_a, box_b)
            count += 1
    result = dict(views=count, exact_masks_and_boxes=True, asset='bus')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
