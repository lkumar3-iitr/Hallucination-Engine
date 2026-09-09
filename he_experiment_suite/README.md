# Hallucination Engine Experiment Suite

`he_experiment_suite_v1` is the reproducible entry point for final HE
experiments. It coordinates existing validated modules without duplicating
their implementations:

- `ScenarioGenerator/`: semantic and resolved scenario generation.
- `he_renderer/`: `he_sprite_renderer_v1` rendering and compositing.
- `driving_models/`: TCP, NEAT, CIL++, and AIM-MT adapters and runtime.
- `important_curated_scenarios/`: frozen evaluation suites.

CARLA must already be running for recording and closed-loop commands. Binary
sprite banks and model checkpoints remain external to Git.

## Preflight

```powershell
python -m he_experiment_suite validate `
  --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets

python -m he_experiment_suite test
```

## Record A Matched Pair

The suite forwards recorder arguments unchanged. The CARLA condition creates
the physical reference; the HE condition uses the frozen production renderer.

```powershell
python -m he_experiment_suite pair --condition carla -- `
  --resolved he_renderer/scenarios/static_left_tesla_ego_overtakes_200.resolved_v2.json `
  --output-dir outputs/final/tesla_carla --actor-id left_tesla --max-frames 201

python -m he_experiment_suite pair --condition he -- `
  --resolved he_renderer/scenarios/static_left_tesla_ego_overtakes_200.resolved_v2.json `
  --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production `
  --manifest he_renderer/manifests/tesla_sidepass_asset_manifest_v1.json `
  --reference-dir outputs/final/tesla_carla `
  --output-dir outputs/final/tesla_he --actor-id left_tesla --max-frames 201
```

## Final Evaluation Campaign

The campaign builds and validates the frozen paper suite, then runs TCP, NEAT,
CIL++, and AIM-MT under matched CARLA and HE conditions. Its default is
explicitly `he_sprite_renderer_v1`; legacy `v1` and `v2` remain available only
for controlled baselines:

```powershell
python -m he_experiment_suite campaign -- `
  --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production `
  --output-root driving_models/outputs/final_evaluation_v1 `
  --device cuda --continue-on-error
```

Run the command once with `--dry-run` before the full campaign. Camera adapters
must supply synchronized RGB, exact camera world transforms, image dimensions,
and FOV. Intrinsics alone do not establish geometric equivalence.

## Reproducibility Boundary

Git contains code, configuration, semantic and resolved scenarios, manifests,
tests, and evaluation logic. External releases contain sprite banks, selector
tables, visual-hull caches, and model checkpoints. Generated frames, videos,
masks, logs, and metrics belong under ignored output directories.
