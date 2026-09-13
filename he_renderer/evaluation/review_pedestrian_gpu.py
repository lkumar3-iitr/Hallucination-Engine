"""Matched-input continuous review on a recorded clean road, not CARLA GT."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'driving_models/common'):
    sys.path.insert(0, str(path))
import carla
import cv2
import numpy as np
from he_asset_registry_v1 import HEAssetRegistry
from he_renderer.calibrated.compositor import HECalibratedCompositor


def transform(row):
    return carla.Transform(carla.Location(**{k:row[k] for k in ('x','y','z')}),
                           carla.Rotation(**{k:row[k] for k in ('pitch','yaw','roll')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--background',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    setup=json.loads((args.background/'setup.json').read_text())
    frames=[json.loads(line) for line in (args.background/'frames.jsonl').read_text().splitlines() if line]
    registry=HEAssetRegistry(args.asset_root,manifest_path=ROOT/'he_renderer/manifests/paper_assets_manifest_v2.json')
    engines=[HECalibratedCompositor(),HECalibratedCompositor(pedestrian_gpu=True)]
    width,height=640,360
    summary=[]
    for mixed in (False,True):
        name='bus_pedestrian' if mixed else 'pedestrian'
        cap=cv2.VideoCapture(str(args.background/'background_only.mp4'))
        writer=cv2.VideoWriter(str(args.output/f'{name}_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                               setup['fps'],(width*2,height+64))
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError('Video open failed')
        rows=[]
        try:
            for index,row in enumerate(frames):
                ok,bg=cap.read()
                if not ok: raise RuntimeError(f'Missing background frame {index}')
                bg=cv2.cvtColor(cv2.resize(bg,(width,height)),cv2.COLOR_BGR2RGB)
                pose=row['actor_transform']
                actors=[]
                for key,offset in ([('pedestrian.person_01',0.),('vehicle.bus_01',12.)] if mixed else [('pedestrian.person_01',0.)]):
                    asset=registry.resolve(key)
                    radians=np.deg2rad(pose['yaw'])
                    actors.append(SimpleNamespace(actor_id=key,asset_key=key,carla_blueprint=asset.carla_blueprint,
                        world_x=pose['x']+offset*np.cos(radians),world_y=pose['y']+offset*np.sin(radians),
                        world_z=.05-asset.physical_bbox.local_bottom_z_m,world_yaw_deg=pose['yaw'],
                        physical_dimensions=asset.physical_bbox,he_view_matrix_csv=str(asset.view_matrix_csv),
                        he_close_view_matrix_csvs=asset.close_view_matrix_csvs))
                outputs=[engine.render(bg,transform(row['camera_transform']),actors,width,height,setup['camera']['fov_deg']) for engine in engines]
                error=int(np.abs(outputs[0].rgb.astype(np.int16)-outputs[1].rgb.astype(np.int16)).max())
                panel=np.zeros((height+64,width*2,3),np.uint8)
                for j,result in enumerate(outputs):
                    panel[64:,j*width:(j+1)*width]=cv2.cvtColor(result.rgb,cv2.COLOR_RGB2BGR)
                cv2.putText(panel,'Default reference',(10,22),0,.6,(255,255,255),1)
                cv2.putText(panel,'GPU pedestrian candidate',(width+10,22),0,.6,(255,255,255),1)
                cv2.putText(panel,f'Frame {index} | RGB max difference {error} | matched inputs; no scene depth / no CARLA target GT',
                            (10,48),0,.5,(255,255,255),1)
                writer.write(panel)
                if index in (0,25,40,45,50,55,60,75,100,200):
                    cv2.imwrite(str(args.output/f'{name}_{index:03}.png'),panel)
                rows.append(dict(frame=index,max_rgb=error,actors=[[dict(id=r.actor_id,visible=r.rendered,metadata=r.he_metadata)
                    for r in output.actor_results] for output in outputs]))
                if index%50==0: print(name,index,'RGB',error,flush=True)
                if error>1: raise RuntimeError('RGB gate failed')
        finally:
            cap.release()
            writer.release()
            (args.output/f'{name}_frames.json').write_text(json.dumps(rows,indent=2))
        verify=cv2.VideoCapture(str(args.output/f'{name}_comparison.mp4'))
        decoded=0
        while True:
            ok,_=verify.read()
            if not ok: break
            decoded+=1
        verify.release()
        if decoded!=len(frames): raise RuntimeError('Output frame count mismatch')
        summary.append(dict(sequence=name,frames=decoded,max_rgb=max(r['max_rgb'] for r in rows)))
    (args.output/'summary.json').write_text(json.dumps(dict(results=summary,
        scope='Synthetic asset substitution on recorded clean overtake camera trajectory; 640x360; no scene depth; not Scenario08 or physical GT',
        background=str(args.background)),indent=2))
    print(summary,flush=True)


if __name__=='__main__': main()
