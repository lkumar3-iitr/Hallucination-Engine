"""Run the paired recorder with the production HE sprite renderer."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "driving_models" / "common"
for path in (ROOT, COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from he_renderer.compositor import HESpriteRendererCompositor
import record_scenario_carla_he_pair_v3 as recorder


def main():
    args = recorder.build_parser().parse_args()
    if args.condition != "he":
        raise ValueError("Use this launcher for --condition he; use the original recorder for CARLA")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    recorder.HEMultiActorCompositorV1 = HESpriteRendererCompositor
    recorder.HEMultiActorCompositorV2 = HESpriteRendererCompositor
    recorder.main()
    setup_path = output / "setup.json"
    setup = json.loads(setup_path.read_text())
    setup["he_renderer_backend"] = "he_sprite_renderer_v1"
    setup["runtime_ground_truth_used"] = False
    setup_path.write_text(json.dumps(setup, indent=2))


if __name__ == "__main__":
    main()
