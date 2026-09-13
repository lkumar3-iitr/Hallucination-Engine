"""Plan or run the migrated renderer's first ClearNoon paired smoke test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    scenario='long_08_bus_occluded_pedestrian_001'
    suite=ROOT/'ScenarioGenerator/outputs/paper_safety_suite_v1'
    files=[suite/'resolved'/f'{scenario}.resolved_v2.json',suite/'environment'/f'{scenario}.environment_v1.json',
           ROOT/'driving_models/common/outputs/town10_spawn10_to45_route_v1/route.csv']
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    commands=[]
    for condition in ('carla','he'):
        destination=args.output_root.resolve()/condition
        if destination.exists():
            raise FileExistsError(f'Refusing to reuse old condition output: {destination}')
        commands.append([sys.executable,str(ROOT/'he_renderer/runners/run_calibrated_closed_loop.py'),
            '--model','neat','--condition',condition,'--resolved',str(files[0]),
            '--asset-root',str(args.asset_root.resolve()),'--environment-json',str(files[1]),
            '--route-metrics-csv',str(files[2]),'--weather-preset','ClearNoon',
            '--metric-actor-id','hidden_pedestrian','--event-start-s','12','--event-source-start-s','12',
            '--trigger-route-progress-m','55','--trigger-gated-actor-id','hidden_pedestrian',
            '--spawn-index','10','--destination-index','27','--max-frames','801','--save-video',
            '--output-root',str(destination)])
    print(json.dumps(dict(renderer='he_calibrated_renderer_v2',commands=commands,
        execute=args.execute,comparison='closed-loop behavior; not matched-pose image fidelity'),indent=2))
    if args.execute:
        for command in commands:
            subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    main()
