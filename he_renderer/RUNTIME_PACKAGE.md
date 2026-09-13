# Runtime Package and Coverage Campaign

Built 2026-09-12 at `artifacts/he_runtime_package_v1`.
Asset payload: 3,814,306,057 bytes (3.814 GB, 3.552 GiB).
Original banks were copied, not modified or deleted. Every copied payload
file was SHA-256 verified; INVENTORY.json records sizes and hashes.

The package holds native RGBA and metadata for all four assets, compact
calibrated vehicle close banks and targeted pedestrian close banks. Manifests
use package-relative paths. Raw RGB capture products and historical full close
banks are excluded. It includes a source snapshot, not an installed environment:
CARLA, Python/CUDA/CuPy, model adapters/checkpoints and scenario inputs remain
external dependencies. Generated caches are rebuilt or reused by workspace code.

Validation with package paths:
- `artifacts/runtime_package_vehicle_gate_v1`: 36 mixed-vehicle contract cases.
- `artifacts/runtime_package_pedestrian_gate_v1`: 512 pedestrian reference cases.
- 37 unit tests passed before campaign launch.

## Background Campaign

Driver: `evaluation/run_runtime_campaign.py`.
Output: `artifacts/runtime_campaign_v1`.
`STATUS.json` contains the current PID, job, completion count and errors.
`PLAN.json` lists all commands and frozen input/source hashes.
Outer logs: `artifacts/runtime_campaign_v1.stdout.log` and `.stderr.log`.
Per-job logs: `job_000.log`, etc. Individual runs include source receipts,
renderer metadata, metrics CSVs and videos.

Eight full-length Scenario08 smoke runs (four models x two conditions) gate
the following 160-run coverage sweep (20 cases x four models x two conditions).
ClearNoon only, 801 frames per run, one repetition. This is development-suite
coverage, not a held-out evaluation or sufficient statistical equivalence study.
Scenario08 retains the validated hidden-pedestrian trigger gate. Other cases
retain their declared trigger/source-time settings. Collision-control labels
are not outcome assertions.

The driver stops on process failure, incomplete frames/camera logs, missing
depth/candidate metadata, or changed source/input hashes. Driving stops,
collisions and completed routes are retained without selecting favorable runs.
No automatic simulator restart or retry is attempted. Keep CARLA running and
avoid source/scenario edits during the campaign. Consult STATUS.json before
assuming it is still progressing. The detached process may stop if the machine
sleeps, the session ends, or CARLA fails.

Reproduce into a NEW output directory:

```powershell
conda run --no-capture-output -n he_neat python he_renderer/evaluation/run_runtime_campaign.py --package he_renderer/artifacts/he_runtime_package_v1 --output he_renderer/artifacts/my_campaign --execute
```

Omit `--execute` to generate only the plan. This driver intentionally refuses
to reuse output directories; interrupted runs must be audited before resuming.
