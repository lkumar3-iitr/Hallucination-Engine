"""Run version 2 with source provenance and per-actor renderer metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'driving_models/common'):
    sys.path.insert(0,str(path))
from he_renderer.calibrated.compositor import HECalibratedCompositor
import generic_he_closed_loop_runner_v1 as runner


def main():
    if '--help' in sys.argv:
        runner.main()
        return
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--he-renderer-version',default='he_calibrated_renderer_v2')
    settings,_=parser.parse_known_args()
    if settings.he_renderer_version!='he_calibrated_renderer_v2':
        raise ValueError('This entry point runs he_calibrated_renderer_v2 only')
    if '--he-renderer-version' not in sys.argv:
        sys.argv.extend(['--he-renderer-version','he_calibrated_renderer_v2'])
    if not any(a=='--manifest' or a.startswith('--manifest=') for a in sys.argv):
        sys.argv.extend(['--manifest',str(ROOT/'he_renderer/manifests/paper_assets_manifest_v2.json')])
    output=settings.output_root.resolve()
    output.mkdir(parents=True,exist_ok=False)
    sources=sorted(p for p in (ROOT/'he_renderer').rglob('*')
                   if p.suffix in ('.py','.cu','.json','.txt')
                   and not set(p.relative_to(ROOT/'he_renderer').parts)&{'assets','runtime','cache','artifacts','frozen','__pycache__'})
    sources.append(ROOT/'driving_models/common/generic_he_closed_loop_runner_v1.py')
    with zipfile.ZipFile(output/'renderer_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path,str(path.relative_to(ROOT)))
    receipt=dict(renderer='he_calibrated_renderer_v2',arguments=sys.argv[1:],
                 pedestrian_gpu='--he-pedestrian-gpu' in sys.argv,
                 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                 external_assets_required=True)
    (output/'RENDERER_RUN.json').write_text(json.dumps(receipt,indent=2))
    with (output/'renderer_metadata.jsonl').open('w') as log:
        HECalibratedCompositor.diagnostic_log=log
        try:
            runner.main()
        finally:
            HECalibratedCompositor.diagnostic_log=None


if __name__=='__main__':
    main()
