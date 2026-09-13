# Calibrated Renderer Version 2

Version: `he_calibrated_renderer_v2`. Migration date: 2026-09-12.
Offline migration gates passed; live scenario smoke is pending simulator restart.

## Production Layout

- `calibrated/`: calibrated close geometry, visual hull, exact GPU native
  projection/raster/scoring, native warp and device handoff, depth compositing.
- `calibrated/compositor.py`: `HECalibratedCompositor`, existing multi-actor
  result contract, RGB/BGR conversion, per-actor metadata and legacy pedestrian.
- `manifests/calibrated_renderer_v2.json`: calibrated close-bank registration.
- `manifests/paper_assets_manifest_v2.json`: native vehicle banks plus the
  retained pedestrian close banks. Vehicle legacy close banks are not required.
- `assets/calibrated_close/{bus,tesla,patrol}`: local copies of compact banks.
- `runtime/python`, `runtime/cuda_headers`: relocated workspace CuPy/runtime.
- `cache/calibrated_v2`, `cache/cupy`: regenerable local caches.

The new implementation has no imports or runtime paths into the research
directory. The old `renderer.py`, `compositor.py`, and `run_pair.py` remain
available explicitly as version-1 baselines. The package VERSION names v2,
but old named API classes intentionally retain their old behavior.
General legacy runner defaults have not been silently changed. Select v2
explicitly or use its dedicated entry points below.

Native RGBA banks remain external under the supplied `--asset-root`.
The three copied compact banks total 215,541,266 bytes across 1,132 files,
all verified byte-for-byte. No source assets were deleted. Banks, runtime
binaries and caches are ignored by Git; code checkout alone is not a complete
deployment. Provision the manifest paths and matching CUDA/PyTorch/CuPy
dependencies before running elsewhere. Requirements are not a locked environment.

## Retained Algorithm

Exact native selection, guarded double-precision GPU projection, integer GPU
rasterization, resident native warp, calibrated close first-hit sampling,
coverage-boundary blending and native-image clipping are retained. Resident
scoring and distilled/temporal lookup shortcuts are disabled in the v2
compositor. The implementation retains CPU reference fallbacks at numerical
ambiguities. No target masks or target RGB are runtime selector inputs.

Scene depth is passed directly through the generic closed-loop runner's v2
entry, using its existing synchronized depth sensors. The old v1 depth
restriction remains for that baseline. No process-local AST rewriting is used.
Pedestrians retain their existing legacy backend/depth approximation, explicitly
logged as `asset_backend=legacy_pedestrian`; calibrated pedestrian is not enabled.
Unregistered vehicle blueprints fail rather than silently selecting a baseline.
HE-to-HE per-pixel z-buffering and physical virtual-actor collision response
remain unsupported. Geometric overlap and closed-loop behavior are separate metrics.

## Gates

- 180 full-render migration comparisons: RGB, alpha, masks, bounds and metadata
  exact, except the intentional version identity change. Native/close/transition,
  three assets, two resolutions, three yaws, depth present/absent.
- 36 mixed-actor production-adapter cases: RGB equals direct production-core
  composition, ordering/version metadata consistent, depth enabled where supplied.
- Standalone production import resolves the relocated runtime without loading
  the research package. New production modules contain no research naming.
- 33 production unit tests pass, including migrated depth, cache, raster,
  projection guard, tie and GPU sampling tests.
- Dedicated closed-loop launcher help and Scenario 08 dry-run validated.

Migration diagnostic timing remains comparable: one-bus native about 302-311
FPS, transition 297 FPS and close 977 FPS in a short warm paired test. This
is a migration regression check, not new paper-grade throughput evidence.
All tests are finite and offline. No live CARLA run or pedestrian regression
scenario has been completed after migration, so full campaign acceptance is pending.

## Evaluation Entry Points

First inspect the Scenario 08 ClearNoon physical/HE plan:

```powershell
conda run --no-capture-output -n he_neat python he_renderer/runners/evaluate_scenario08.py --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --output-root he_renderer/artifacts/evaluation_v2_scenario08
```

After CARLA restarts, append `--execute`. Both conditions use fresh output
directories and the same scenario/environment/trigger parameters. The 801-frame
NEAT pair is a closed-loop smoke test, not a matched-pose fidelity comparison.
Inspect video and logs before the wider campaign; retain stopped outcomes.

For other live scenarios use `runners/run_calibrated_closed_loop.py` with
existing model/scenario arguments. It forces the v2 identity, defaults to the
v2 asset manifest, and records source hashes/archive and per-actor metadata.
Outputs must be new directories. External bank files are not included in the
source archive. Generic/campaign runners also accept
`--he-renderer-version he_calibrated_renderer_v2` and the v2 manifest explicitly;
use the dedicated launcher when source receipts are required.

For matched replay use `runners/run_calibrated_pair.py`; the existing pair
recorder does not supply scene depth, and setup records that limitation.
Existing comparison and metric tools continue to consume the same result schema.

```powershell
$env:PYTHONPATH='D:/HallucinationEngine;D:/HallucinationEngine/driving_models/common'
conda run --no-capture-output -n he_neat python -m unittest discover -s he_renderer/tests -p test_*.py
conda run --no-capture-output -n he_neat python he_renderer/evaluation/validate_calibrated_contract.py --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --output he_renderer/artifacts/my_v2_contract
```

No commit/push performed. Existing unrelated workspace changes were preserved.
