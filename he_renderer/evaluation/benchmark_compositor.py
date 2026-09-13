"""Warm offline compositor throughput, including the legacy pedestrian path."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'driving_models/common'):
    sys.path.insert(0, str(path))

import carla
import numpy as np
import torch
from he_asset_registry_v1 import HEAssetRegistry
from he_renderer.calibrated.compositor import HECalibratedCompositor


def camera_specs(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'camera_specs']
    if len(methods) != 1:
        raise ValueError('Expected one camera_specs method')
    returns = [n for n in methods[0].body if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, (ast.List, ast.Tuple)):
        raise ValueError('Expected literal camera list')
    result = []
    for call in returns[0].value.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'CameraSpec' or call.args:
            raise ValueError('Expected literal CameraSpec keywords')
        result.append({k.arg: ast.literal_eval(k.value) for k in call.keywords})
    return result


def clear_queries(compositor):
    for pipeline in compositor.gpu_engines.values():
        pipeline.native.selector.clear()
    for engine in compositor.bus_engines.values():
        engine.close.hull._intersection_cache = None
    for hull in compositor.depth_hulls.values():
        hull._intersection_cache = None
    for engine in compositor.base.engines.values():
        if hasattr(engine.selector, 'clear'):
            engine.selector.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=92184)
    parser.add_argument('--pedestrian-gpu', action='store_true')
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error('At least 10 repeats required')
    check = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process CarlaUE4* -ErrorAction SilentlyContinue).Count'], capture_output=True, text=True, check=True)
    if check.stdout.strip() != '0':
        raise RuntimeError('Close CARLA before this offline benchmark')
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {}
    for name, folder in [('tcp', 'TCP'), ('neat', 'NEAT'), ('cilpp', 'CILPP'), ('aimmt', 'AIMMT')]:
        path = ROOT/'driving_models'/folder/f'{name}_adapter_v1.py'
        profiles[name] = dict(specs=camera_specs(path), source=str(path.relative_to(ROOT)),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
    (args.output/'environment.json').write_text(json.dumps(dict(gpu=gpu.stdout, carla_count=0,
        python=sys.version, numpy=np.__version__, torch=torch.__version__), indent=2))
    with zipfile.ZipFile(args.output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT/'he_renderer').rglob('*'):
            if path.suffix in ('.py', '.cu', '.json') and not set(path.relative_to(ROOT/'he_renderer').parts) & {'assets', 'cache', 'runtime', 'artifacts', 'frozen'}:
                archive.write(path, path.relative_to(ROOT))
        for path in (ROOT/'driving_models/common').glob('*.py'):
            archive.write(path, path.relative_to(ROOT))
        for profile in profiles.values():
            archive.write(ROOT/profile['source'], profile['source'])
    registry = HEAssetRegistry(args.asset_root, manifest_path=ROOT/'he_renderer/manifests/paper_assets_manifest_v2.json')
    keys = dict(bus='vehicle.bus_01', tesla='vehicle.passenger_01', patrol='vehicle.passenger_02', pedestrian='pedestrian.person_01')
    groups = [('bus',), ('pedestrian',), ('bus', 'pedestrian'), ('bus', 'tesla', 'patrol', 'pedestrian')]
    compositor = HECalibratedCompositor(pedestrian_gpu=args.pedestrian_gpu)
    jobs = [(model, offset, group) for model in profiles for offset in (30., 12., 6., 0.) for group in groups]
    np.random.default_rng(args.seed).shuffle(jobs)
    rows = []
    for model, offset, group in jobs:
        actors = []
        for i, name in enumerate(group):
            asset = registry.resolve(keys[name])
            bounds = asset.physical_bbox
            if bounds is None:
                raise ValueError(f'Missing physical bounds: {name}')
            actors.append(SimpleNamespace(actor_id=name, asset_key=asset.key, carla_blueprint=asset.carla_blueprint,
                world_x=float(i*10), world_y=-4.2, world_z=0., world_yaw_deg=0.,
                physical_dimensions=SimpleNamespace(length_m=bounds.length_m, width_m=bounds.width_m, height_m=bounds.height_m),
                he_view_matrix_csv=str(asset.view_matrix_csv), he_close_view_matrix_csvs=asset.close_view_matrix_csvs))
        views = []
        for spec in profiles[model]['specs']:
            camera = carla.Transform(carla.Location(x=-offset+spec['x_m'], y=spec['y_m'], z=spec['z_m']),
                carla.Rotation(pitch=spec['pitch_deg'], yaw=spec['yaw_deg'], roll=spec['roll_deg']))
            bg = np.full((spec['height'], spec['width'], 3), 80, np.uint8)
            depth = np.full(bg.shape[:2], 1000., dtype=np.float32)
            depth[:, spec['width']//3:spec['width']//2] = .5
            views.append((spec, camera, bg, depth))

        def render():
            clear_queries(compositor)
            results = [compositor.render(bg, camera, actors, spec['width'], spec['height'], spec['fov_deg'], depth)
                       for spec, camera, bg, depth in views]
            return np.stack([r.rgb for r in results]), results

        expected, details = render()
        for _ in range(10):
            render()
        samples = []
        maximum = 0
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual, _ = render()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
            maximum = max(maximum, int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max()))
        rows.append(dict(model=model, offset=offset, assets=group, cameras=len(views), raw_ms=samples,
            mean_ms=float(np.mean(samples)), p95_ms=float(np.percentile(samples,95)), sets_per_s=1000/float(np.mean(samples)),
            max_repeat_rgb=maximum, actor_metadata=[[dict(actor=r.actor_id, rendered=r.rendered, metadata=r.he_metadata)
                for r in result.actor_results] for result in details]))
        (args.output/'summary.json').write_text(json.dumps(dict(results=rows, profiles=profiles, seed=args.seed,
            repeats=args.repeats, warmups=10, renderer='he_calibrated_renderer_v2', pedestrian_gpu=args.pedestrian_gpu,
            scope='CPU RGB/depth through production compositor and CPU camera stack; no model, simulator, video, file logging or cold setup',
            quality='Same-implementation repeatability only; not an independent fidelity gate'), indent=2))
        print(model, offset, group, round(rows[-1]['sets_per_s'], 2), 'sets/s', 'repeat RGB', maximum, flush=True)
        if maximum:
            raise RuntimeError('Repeatability gate failed')


if __name__ == '__main__':
    main()
