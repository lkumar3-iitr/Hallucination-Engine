import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    project_root = Path(".")
    base_config_path = project_root / "configs" / "v1_straight.yaml"
    temp_config_dir = project_root / "configs" / "generated_runs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    runs = [
        {"name": "v1_straight_run01", "seed": 41, "num_samples": 10000},
        {"name": "v1_straight_run02", "seed": 42, "num_samples": 10000},
        {"name": "v1_straight_run03", "seed": 43, "num_samples": 10000},
    ]

    manifest = {
        "base_config": str(base_config_path),
        "runs": runs,
    }

    manifest_path = project_root / "outputs" / "generate_v1_30k_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Base config: {base_config_path}")
    print(f"[INFO] Manifest saved to: {manifest_path}")
    print(f"[INFO] Total planned samples: {sum(r['num_samples'] for r in runs)}")

    for i, run in enumerate(runs, start=1):
        run_name = run["name"]
        run_seed = int(run["seed"])
        run_num_samples = int(run["num_samples"])

        run_cfg = copy.deepcopy(base_cfg)

        # Set per-run output directory.
        if "output" not in run_cfg:
            run_cfg["output"] = {}
        run_cfg["output"]["dataset_dir"] = f"dataset/{run_name}"

        # Optional: store ego spawn index in metadata if you later decide to use it.
        # For now we rely on random seed to vary ego spawn.
        if "carla" not in run_cfg:
            run_cfg["carla"] = {}

        temp_cfg_path = temp_config_dir / f"{run_name}.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

        cmd = [
            sys.executable,
            "data_generation/collect_carla_bbox_dataset.py",
            "--config",
            str(temp_cfg_path),
            "--num-samples",
            str(run_num_samples),
            "--seed",
            str(run_seed),
        ]

        print()
        print("=" * 80)
        print(f"[RUN {i}/{len(runs)}] {run_name}")
        print(f"[INFO] Seed        : {run_seed}")
        print(f"[INFO] Samples     : {run_num_samples}")
        print(f"[INFO] Config      : {temp_cfg_path}")
        print(f"[INFO] Output dir  : dataset/{run_name}")
        print(f"[INFO] Command     : {' '.join(cmd)}")
        print("=" * 80)
        print()

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[ERROR] Run failed: {run_name} (return code {result.returncode})")
            sys.exit(result.returncode)

        print()
        print(f"[DONE] Completed {run_name}")
        print()

    print("=" * 80)
    print("[DONE] All runs completed successfully.")
    print("[DONE] Generated datasets:")
    for run in runs:
        print(f"  - dataset/{run['name']}")
    print("[DONE] Total requested samples: 30000")
    print("=" * 80)


if __name__ == "__main__":
    main()