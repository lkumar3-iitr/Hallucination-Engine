"""Matched replay using calibrated version 2; recorder supplies no scene depth."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'driving_models/common'):
    sys.path.insert(0,str(path))
from he_renderer.calibrated.compositor import HECalibratedCompositor
import record_scenario_carla_he_pair_v3 as recorder


def main():
    args=recorder.build_parser().parse_args()
    if args.condition!='he':
        raise ValueError('Use the physical recorder for the CARLA condition')
    output=Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    recorder.HEMultiActorCompositorV1=HECalibratedCompositor
    recorder.HEMultiActorCompositorV2=HECalibratedCompositor
    recorder.main()
    path=output/'setup.json'
    setup=json.loads(path.read_text())
    setup.update(he_renderer_backend='he_calibrated_renderer_v2',runtime_ground_truth_used=False,
                 scene_depth_enabled=False)
    path.write_text(json.dumps(setup,indent=2))


if __name__=='__main__':
    main()
