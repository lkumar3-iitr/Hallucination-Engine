"""Run the ASTRA candidate without editing or promoting production code."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"driving_models/common"))
sys.path.insert(0, str(ROOT.parent))

from astra.calibrated_compositor import CalibratedCompositor
import generic_he_closed_loop_runner_v1 as runner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-bank", type=Path, default=CalibratedCompositor.bank_path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--candidate-scene-depth", action="store_true")
    candidate, remaining = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--output-root", type=Path, required=True)
    check.add_argument("--he-renderer-version", choices=["he_sprite_renderer_v1"], required=True)
    settings, _ = check.parse_known_args(remaining)
    output = settings.output_root.resolve()
    if ROOT not in output.parents:
        raise ValueError("Candidate output must remain inside astra")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"renderer": "astra_calibrated_hull_candidate_v1", "candidate_bank": str(candidate.candidate_bank.resolve()),
               "scene_depth_enabled": candidate.candidate_scene_depth,
               "runner_compatibility_alias": "he_sprite_renderer_v1", "arguments": remaining,
               "production_promoted": False}
    (output/"CANDIDATE_RUN.json").write_text(json.dumps(receipt, indent=2))
    CalibratedCompositor.bank_path = candidate.candidate_bank
    if candidate.candidate_manifest:
        manifest = candidate.candidate_manifest.resolve()
        entries = json.loads(manifest.read_text())["assets"]
        CalibratedCompositor.bank_paths = {
            entry["blueprint"]: (manifest.parent/entry["close_bank"]).resolve()
            for entry in entries if entry.get("enabled", True)}
        receipt["candidate_manifest"] = str(manifest)
        receipt["candidate_banks"] = {key: str(value) for key, value in CalibratedCompositor.bank_paths.items()}
        (output/"CANDIDATE_RUN.json").write_text(json.dumps(receipt, indent=2))
    original = runner.HESpriteRendererCompositor
    original_main = runner.main
    if candidate.candidate_scene_depth:
        from astra.scene_depth import enable_runner_depth
        enable_runner_depth(runner)
    runner.HESpriteRendererCompositor = CalibratedCompositor
    sys.argv = [sys.argv[0], *remaining]
    try:
        with (output/"renderer_metadata.jsonl").open("w") as log:
            CalibratedCompositor.diagnostic_log = log
            runner.main()
    finally:
        CalibratedCompositor.diagnostic_log = None
        runner.HESpriteRendererCompositor = original
        runner.main = original_main


if __name__ == "__main__":
    main()
