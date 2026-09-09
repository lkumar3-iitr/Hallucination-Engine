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

## Evaluation Gate

Before launching the full campaign, run one short matched CARLA/HE smoke case
for each of TCP, NEAT, CIL++, and AIM-MT with its native camera configuration.
Inspect the composited camera videos and confirm that run metadata records
`he_sprite_renderer_v1`. This gate checks checkpoint availability, adapter
camera calibration, synchronization, and live CARLA integration; it is not a
new renderer-development phase.

Depth-based scene occlusion remains outside this release and should be added as
a separately measured compositor revision after the non-occlusion evaluation
baseline is frozen.
