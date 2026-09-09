# HE Experiment Suite Release Status

Status date: 2026-09-09

## Frozen Components

- Renderer: `he_sprite_renderer_v1`
- Suite: `he_experiment_suite_v1`
- Source checkpoint before suite packaging: `f5e019a8d2127c76ffbdefe0a2e38b27786e6a5f`
- Asset release: `he_assets_sidepass_v1`
- Assets: Tesla Model 3, Fuso Rosa bus, Nissan Patrol, pedestrian 0001
- Selection: guarded distilled selector with exact visual-hull fallback
- Near field: manifest-selected targeted Cartesian close banks

## Reproduced Evidence

- Complete Tesla, bus, patrol, and pedestrian static-adversary side passes.
- Matched physical-CARLA and HE comparison videos and geometry metrics.
- Complete 40-second three-camera NEAT replay.
- Release asset paths and identity-critical SHA-256 hashes validated.
- 15 renderer tests and 4 suite tests passed.
- 47 ScenarioGenerator tests passed.
- Frozen 20-scenario paper suite rebuilt and validated: 15 collision-free and
  5 collision-required controls.
- Four-model, two-condition campaign expanded to 160 commands in dry-run mode.
- TCP, NEAT, CIL++, and AIM-MT each completed the 801-frame ClearNoon
  `signalized_lead_follow_001` scenario under CARLA and HE.
- Four synchronized long-run comparison videos passed critical-frame review.

## Evaluation Gate

The live integration gate is complete for TCP, NEAT, CIL++, and AIM-MT. The
next stage is the repeated full evaluation campaign. Preserve per-repeat seeds
and report confidence intervals; do not treat the single readiness runs as the
final behavioral estimates. See `CLEAR_WEATHER_READINESS_RESULTS.md`.

Depth-based scene occlusion remains outside this release and should be added as
a separately measured compositor revision after the non-occlusion evaluation
baseline is frozen.
