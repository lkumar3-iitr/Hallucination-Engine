"""Run the existing paired HE recorder with ASTRA injected in this process only."""
import json
from pathlib import Path

from compositor import AstraCompositor
import record_scenario_carla_he_pair_v3 as recorder


def main():
    args = recorder.build_parser().parse_args()
    if args.condition != "he":
        raise ValueError("Use this launcher for --condition he; use the original recorder for CARLA")
    output = Path(args.output_dir).resolve()
    astra_root = Path(__file__).resolve().parent
    if not output.is_relative_to(astra_root):
        raise ValueError("ASTRA experimental recorder outputs must be inside astra/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    recorder.HEMultiActorCompositorV1 = AstraCompositor
    recorder.HEMultiActorCompositorV2 = AstraCompositor
    recorder.main()
    setup_path = output / "setup.json"
    setup = json.loads(setup_path.read_text())
    setup["he_renderer_backend"] = "astra_passby_v1"
    setup["astra_runtime_GT"] = False
    setup_path.write_text(json.dumps(setup, indent=2))


if __name__ == "__main__":
    main()
