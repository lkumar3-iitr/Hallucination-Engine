"""Compare opt-in pedestrian acceleration against the untouched default path."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile
import cProfile
import pstats

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'driving_models/common'):
    sys.path.insert(0, str(path))
import carla
import numpy as np
import torch
from he_asset_registry_v1 import HEAssetRegistry
from he_renderer.calibrated.compositor import HECalibratedCompositor
from he_renderer.evaluation.benchmark_compositor import camera_specs, clear_queries


def reset(compositor):
    clear_queries(compositor)
    for engine in compositor.base.engines.values():
        if hasattr(engine.selector, 'clear'):
            engine.selector.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--asset-manifest',type=Path,default=ROOT/'he_renderer/manifests/paper_assets_manifest_v2.json')
    parser.add_argument('--calibrated-manifest',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.output/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT/'he_renderer').rglob('*'):
            if path.suffix in ('.py', '.cu', '.json') and not set(path.relative_to(ROOT/'he_renderer').parts) & {'assets','artifacts','runtime','cache','frozen'}:
                archive.write(path, path.relative_to(ROOT))
    registry = HEAssetRegistry(args.asset_root, manifest_path=args.asset_manifest)
    asset = registry.resolve('pedestrian.person_01')
    actor = SimpleNamespace(actor_id='pedestrian', asset_key=asset.key, carla_blueprint=asset.carla_blueprint,
        world_x=0., world_y=-4.2, world_z=0., world_yaw_deg=0., physical_dimensions=asset.physical_bbox,
        he_view_matrix_csv=str(asset.view_matrix_csv), he_close_view_matrix_csvs=asset.close_view_matrix_csvs)
    reference = HECalibratedCompositor(calibrated_manifest=args.calibrated_manifest)
    candidate = HECalibratedCompositor(calibrated_manifest=args.calibrated_manifest,pedestrian_gpu=True)
    rows = []
    profiles = {name: camera_specs(ROOT/'driving_models'/folder/f'{name}_adapter_v1.py')
        for name,folder in [('tcp','TCP'),('neat','NEAT'),('cilpp','CILPP'),('aimmt','AIMMT')]}
    for model,specs in profiles.items():
        for offset in (30.,12.,6.,0.):
            for side in (-1,1):
                actor.world_y = side*4.2
                for yaw in (0.,35.):
                    actor.world_yaw_deg = yaw
                    for spec in specs:
                        camera = carla.Transform(carla.Location(x=-offset+spec['x_m'], z=spec['z_m'], y=spec['y_m']),
                            carla.Rotation(yaw=spec['yaw_deg'],pitch=spec['pitch_deg'],roll=spec['roll_deg']))
                        bg = np.random.default_rng(9).integers(0,256,(spec['height'],spec['width'],3),dtype=np.uint8)
                        for kind in ('none','strip','invalid','foreground'):
                            depth = None
                            if kind != 'none':
                                depth = np.full(bg.shape[:2],1000.,dtype=np.float32)
                                depth[:,spec['width']//3:spec['width']//2] = .5
                                if kind == 'foreground':
                                    depth[:] = .1
                                if kind == 'invalid':
                                    depth[:,:spec['width']//3] = np.nan
                                    depth[:,spec['width']//3:2*spec['width']//3] = -1
                                    depth[:,2*spec['width']//3:] = np.inf
                            outputs = []
                            for engine in (reference,candidate):
                                reset(engine)
                                outputs.append(engine.render(bg,camera,[actor],spec['width'],spec['height'],spec['fov_deg'],depth))
                            error = int(np.abs(outputs[0].rgb.astype(np.int16)-outputs[1].rgb.astype(np.int16)).max())
                            a,b = [dict(o.actor_results[0].he_metadata) for o in outputs]
                            b.pop('performance_backend',None)
                            # Compare public metadata numerically; GPU bounds may differ below 1e-9 px.
                            def equal(x,y):
                                if isinstance(x,dict):
                                    return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
                                if isinstance(x,(list,tuple)):
                                    return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
                                if isinstance(x,(int,float)) and not isinstance(x,bool):
                                    return bool(np.isclose(x,y,rtol=0,atol=1e-9,equal_nan=True))
                                return x==y
                            passed = error<=1 and equal(a,b)
                            alpha_error = 0.
                            mask_changes = 0
                            if kind == 'none' and a['sprite_mode'] == 'view_matrix':
                                actor_tf = carla.Transform(carla.Location(x=actor.world_x,y=actor.world_y,z=actor.world_z),carla.Rotation(yaw=yaw))
                                native = [engine.base.engines[str(asset.asset_dir.resolve())] for engine in (reference,candidate)]
                                layers = [engine.render(bg,actor_tf,camera,spec['fov_deg']) for engine in native]
                                alpha_error = float(np.max(np.abs(layers[0][1]-layers[1][1])))
                                mask_changes = int(np.count_nonzero((layers[0][1]>10/255)!=(layers[1][1]>10/255)))
                                passed = passed and mask_changes == 0 and alpha_error <= 1e-4
                            rows.append(dict(model=model,offset=offset,side=side,actor_yaw=yaw,camera_yaw=spec['yaw_deg'],depth=kind,
                                max_rgb=error,metadata_equal=equal(a,b),alpha_error=alpha_error,mask_changes=mask_changes,
                                sprite_mode=a['sprite_mode'],selection_mode=a.get('selection_mode'),passed=passed))
                            (args.output/'quality.json').write_text(json.dumps(rows,indent=2))
                            if not passed:
                                (args.output/'failure.json').write_text(json.dumps(dict(reference=a,candidate=b),indent=2))
                                raise RuntimeError(f'Quality gate failed: {rows[-1]}')
            print('quality',model,offset,'passed',flush=True)
    if args.quality_only:
        print('PASS',len(rows),'quality views',flush=True)
        return
    # Isolated candidate and reference blocks, order alternated between workloads.
    timing = []
    actor.world_y=-4.2
    actor.world_yaw_deg=0.
    for model,specs in profiles.items():
        for offset in (30.,12.,6.,0.):
            views=[]
            for spec in specs:
                camera=carla.Transform(carla.Location(x=-offset+spec['x_m'],z=spec['z_m'],y=spec['y_m']),carla.Rotation(yaw=spec['yaw_deg']))
                bg=np.full((spec['height'],spec['width'],3),80,np.uint8)
                depth=np.full(bg.shape[:2],1000.,np.float32)
                depth[:,spec['width']//3:spec['width']//2]=.5
                views.append((spec,camera,bg,depth))
            for label,engine in ([('reference',reference),('candidate',candidate)] if len(timing)%4==0 else [('candidate',candidate),('reference',reference)]):
                def render():
                    reset(engine)
                    return np.stack([engine.render(bg,cam,[actor],s['width'],s['height'],s['fov_deg'],depth).rgb for s,cam,bg,depth in views])
                for _ in range(10): render()
                samples=[]
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    render()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
                timing.append(dict(model=model,offset=offset,method=label,raw_ms=samples,fps=1000/float(np.mean(samples))))
                (args.output/'timing.json').write_text(json.dumps(timing,indent=2))
                print(model,offset,label,round(timing[-1]['fps'],2),flush=True)
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(30):
        reset(candidate)
        candidate.render(bg,camera,[actor],spec['width'],spec['height'],spec['fov_deg'],depth)
    profile.disable()
    with (args.output/'candidate_profile.txt').open('w') as stream:
        pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(45)
    print('PASS',len(rows),'quality views',flush=True)


if __name__ == '__main__':
    main()
