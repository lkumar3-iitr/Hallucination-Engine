"""Fail-closed smoke gates followed by one frozen, sequential coverage sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2))
    temporary.replace(path)


def validate_terminal_summary(summary):
    if not summary.get('stop_on_collision'):
        raise RuntimeError('Missing terminal collision policy')
    frames = summary['frames']
    if not isinstance(frames, int) or not 0 <= frames <= 801:
        raise RuntimeError('Invalid frame count')
    event = summary.get('collision_event')
    if summary.get('termination_reason') == 'collision':
        if (not event or event.get('physical_overlap') is not True
                or event.get('criterion') != 'scenario_actor_2d_footprint_overlap'
                or event.get('scenario_frame') != frames
                or event.get('rendered') is not False
                or event.get('model_executed') is not False):
            raise RuntimeError('Invalid collision termination receipt')
    elif frames != 801 or summary.get('termination_reason') != 'horizon' or event is not None:
        raise RuntimeError('Incomplete non-collision run')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    package=args.package.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    suite_path=ROOT/'ScenarioGenerator/outputs/paper_safety_suite_v1/suite_manifest.json'
    suite=json.loads(suite_path.read_text())
    smoke=next(c for c in suite['cases'] if c['scenario_id']=='long_08_bus_occluded_pedestrian_001')
    models=('neat','tcp','cilpp','aimmt')
    jobs=[]
    tracked={suite_path,package/'INVENTORY.json',package/'manifests/assets.json',package/'manifests/calibrated.json'}
    contact_smoke=next(c for c in suite['cases'] if c['scenario_id']=='long_05_crossing_nissan_001')
    for phase,cases in [('smoke',[contact_smoke,smoke]),('coverage',suite['cases'])]:
        for case in cases:
            resolved=(ROOT/'ScenarioGenerator'/case['resolved']).resolve()
            if sha(resolved)!=case['resolved_sha256']: raise RuntimeError(f'Scenario hash mismatch: {resolved}')
            environment=(ROOT/'ScenarioGenerator'/case['environment']).resolve()
            route=(ROOT/suite['route_csv']).resolve()
            tracked.update((resolved,environment,route))
            for model in models:
                for condition in ('carla','he'):
                    destination=output/phase/case['scenario_id']/model/condition
                    command=[sys.executable,str(ROOT/'he_renderer/runners/run_calibrated_closed_loop.py'),
                        '--model',model,'--condition',condition,'--resolved',str(resolved),
                        '--asset-root',str(package/'assets/native'),'--manifest',str(package/'manifests/assets.json'),
                        '--he-calibrated-manifest',str(package/'manifests/calibrated.json'),'--he-pedestrian-gpu',
                        '--environment-json',str(environment),'--route-metrics-csv',str(route),
                        '--weather-preset','ClearNoon','--metric-actor-id',case['metric_actor_id'],
                        '--event-start-s',str(case['event_start_s']),
                        '--event-source-start-s',str(case.get('event_source_start_s',case['event_start_s'])),
                        '--spawn-index','10','--destination-index','27','--max-frames','801','--save-video','--stop-on-collision',
                        '--output-root',str(destination)]
                    if case.get('trigger_route_progress_m') is not None:
                        command+=['--trigger-route-progress-m',str(case['trigger_route_progress_m'])]
                    if case.get('pre_trigger_source_frame') is not None:
                        command+=['--pre-trigger-source-frame',str(case['pre_trigger_source_frame'])]
                    gated=case.get('trigger_gated_actor_ids',[])
                    if case['scenario_id']==smoke['scenario_id']:
                        gated=['hidden_pedestrian']
                    for actor_id in gated: command+=['--trigger-gated-actor-id',actor_id]
                    jobs.append(dict(phase=phase,scenario=case['scenario_id'],model=model,condition=condition,
                        output=str(destination),command=command,status='pending'))
    for folder in (ROOT/'he_renderer',ROOT/'driving_models/common'):
        for path in folder.rglob('*.py'):
            if not set(path.relative_to(folder).parts)&{'artifacts','runtime','cache','assets','frozen','__pycache__'}:
                tracked.add(path)
    frozen={str(p):sha(p) for p in tracked}
    plan=dict(pid=os.getpid(),package=str(package),jobs=jobs,input_hashes=frozen,
        scope='Sixteen contact/integration smoke runs, then 160 coverage runs; terminal on overlap; one repetition',
        scenario08_override='Gate hidden_pedestrian until route trigger, matching accepted smoke setup',
        failure_policy='Stop on infrastructure/completeness failure; retain all driving outcomes')
    save(output/'PLAN.json',plan)
    if not args.execute:
        print('Planned',len(jobs),'jobs')
        return
    status=dict(pid=os.getpid(),state='running',completed=0,total=len(jobs),started=time.time())
    try:
        for index,job in enumerate(jobs):
            for path,expected in frozen.items():
                if sha(Path(path))!=expected: raise RuntimeError(f'Frozen source/input changed: {path}')
            status.update(active=job,completed=index)
            save(output/'STATUS.json',status)
            print(index+1,'/',len(jobs),job['phase'],job['scenario'],job['model'],job['condition'],flush=True)
            log=output/f'job_{index:03}.log'
            with log.open('w') as handle:
                result=subprocess.run(job['command'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Job {index} failed ({result.returncode}); see {log}')
            summaries=list(Path(job['output']).rglob('*_runtime_summary.json'))
            if len(summaries)!=1: raise RuntimeError('Missing runtime summary')
            summary=json.loads(summaries[0].read_text())
            validate_terminal_summary(summary)
            if summary['termination_reason']=='collision':
                receipts=list(Path(job['output']).rglob('collision_event.json'))
                if len(receipts)!=1 or json.loads(receipts[0].read_text())!=summary['collision_event']:
                    raise RuntimeError('Missing or inconsistent collision receipt')
            if job['condition']=='he':
                metadata_path=Path(job['output'])/'renderer_metadata.jsonl'
                records=([json.loads(line) for line in metadata_path.read_text().splitlines()]
                         if metadata_path.exists() else [])
                if len(records)!=summary['frames']*summary['camera_count']: raise RuntimeError('Incomplete camera logs')
                actors=[a for r in records for a in r['actors']]
                if (summary['frames'] and not actors) or any(not a['metadata'].get('scene_occlusion',{}).get('enabled') for a in actors):
                    raise RuntimeError('Missing scene-depth metadata')
                pedestrians=[a for a in actors if a['metadata'].get('asset_backend')=='legacy_pedestrian']
                if any(a['metadata'].get('performance_backend')!='pedestrian_gpu_candidate_v1' for a in pedestrians):
                    raise RuntimeError('Missing pedestrian candidate identity')
            job.update(status='complete',runtime_summary=str(summaries[0]))
            status.update(completed=index+1)
            save(output/'PLAN.json',plan)
            save(output/'STATUS.json',status)
        status.update(state='complete',finished=time.time(),active=None)
    except Exception as exc:
        status.update(state='blocked',error=str(exc),finished=time.time())
        save(output/'STATUS.json',status)
        raise
    save(output/'STATUS.json',status)


if __name__=='__main__': main()
