"""Offline RGB/depth/multi-actor contract gate for calibrated version 2."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'driving_models/common'):
    sys.path.insert(0,str(path))
import carla
import cv2
import numpy as np
from he_renderer.calibrated.compositor import HECalibratedCompositor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibrated-manifest',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    compositor=HECalibratedCompositor(calibrated_manifest=args.calibrated_manifest)
    specs=[('vehicle.mitsubishi.fusorosa','fuso_rosa_bus_native_full_v3'),
           ('vehicle.tesla.model3','tesla_model3_native_full_v3'),
           ('vehicle.nissan.patrol_2021','nissan_patrol_2021_native_full_v3')]
    actors=[SimpleNamespace(actor_id=str(i),asset_key=blueprint,carla_blueprint=blueprint,
        world_x=float(i*10),world_y=-4.2,world_z=0.,world_yaw_deg=0.,
        physical_dimensions=SimpleNamespace(length_m=8.,width_m=2.,height_m=3.),
        he_view_matrix_csv=str(args.asset_root/folder/'view_matrix.csv')) for i,(blueprint,folder) in enumerate(specs)]
    bg=np.random.default_rng(101).integers(0,256,(300,400,3),dtype=np.uint8)
    rows=[]
    for offset in (30.,12.,6.,0.):
        for yaw in (0.,-60.,60.):
            camera=carla.Transform(carla.Location(x=-offset,z=2.3),carla.Rotation(yaw=yaw))
            for depth_kind in ('none','strip','foreground'):
                scene=None if depth_kind=='none' else np.full((300,400),.1 if depth_kind=='foreground' else 1000.)
                if depth_kind=='strip':
                    scene[:,100:180]=.1
                actual=compositor.render(bg,camera,actors,400,300,100,scene)
                expected=cv2.cvtColor(bg,cv2.COLOR_RGB2BGR)
                ordered=sorted(actors,key=lambda a:(a.world_x+offset)**2,reverse=True)
                for actor in ordered:
                    engine=compositor.gpu_engines[str(Path(actor.he_view_matrix_csv).resolve().parent)]
                    transform=carla.Transform(carla.Location(x=actor.world_x,y=actor.world_y))
                    expected,_,_=engine.render(expected,transform,camera,100,scene,include_occlusion_metadata=True)
                np.testing.assert_array_equal(actual.rgb,cv2.cvtColor(expected,cv2.COLOR_BGR2RGB))
                assert [r.actor_id for r in actual.actor_results]==[a.actor_id for a in ordered]
                assert all(r.he_metadata['renderer_backend']=='he_calibrated_renderer_v2' for r in actual.actor_results)
                if scene is not None:
                    assert all(r.he_metadata['scene_occlusion']['enabled'] for r in actual.actor_results)
                rows.append(dict(offset=offset,yaw=yaw,depth=depth_kind,passed=True))
    empty=compositor.render(bg,camera,[],400,300,100)
    np.testing.assert_array_equal(empty.rgb,bg)
    assert not empty.actor_results
    (args.output/'summary.json').write_text(json.dumps(dict(views=len(rows),rows=rows,
        version='he_calibrated_renderer_v2',reference='direct production core; adapter contract only',
        independent_package_import=True),indent=2))
    print('Passed',len(rows),'mixed-actor RGB/depth contract cases')


if __name__=='__main__':
    main()
