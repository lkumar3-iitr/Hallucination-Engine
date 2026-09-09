# Clear-Weather Evaluation Readiness

Run date: 2026-09-09

Renderer: `he_sprite_renderer_v1`

Weather: CARLA `ClearNoon`

## Static Overtake Geometry

The accepted guarded-distilled side-pass comparisons contain 201 frames at
20 FPS. All four assets have complete CARLA/HE visibility agreement.

| Asset | Visible frames | Visibility | BBox IoU | Mask IoU | Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tesla Model 3 | 61 | 1.0000 | 0.9562 | 0.9423 | 0.9654 |
| Fuso Rosa bus | 72 | 1.0000 | 0.9636 | 0.9091 | 0.9402 |
| Nissan Patrol | 62 | 1.0000 | 0.9721 | 0.9468 | 0.9724 |
| Pedestrian 0001 | 52 | 1.0000 | 0.8316 | 0.8060 | 0.8894 |

The pedestrian score remains limited by its static sprite articulation versus
CARLA animation. This is an appearance-model limitation, not a projection or
visibility failure.

## Forty-Second Model Runs

Scenario: `signalized_lead_follow_001`

TCP, NEAT, CIL++, and AIM-MT each completed 801-frame CARLA and HE runs. All
runs used their native camera layouts, loaded their intended checkpoints, and
finished without collision. HE metadata selected `he_sprite_renderer_v1`.

| Model | Cameras | XY MAE (m) | XY P95 (m) | Speed MAE (m/s) | Progress MAE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TCP | 1 | 1.981 | 6.053 | 1.091 | 1.967 |
| NEAT | 3 | 0.470 | 0.981 | 0.271 | 0.470 |
| CIL++ | 3 | 0.643 | 2.370 | 0.555 | 0.533 |
| AIM-MT | 1 | 0.615 | 1.607 | 0.494 | 0.616 |

These values measure one closed-loop matched repeat. They are not final model
confidence intervals. TCP has the largest trajectory separation and must be
covered by the planned repeated-trial evaluation rather than interpreted from
this single run.

## Artifacts

Long-run outputs and comparison videos:

```text
driving_models/outputs/final_readiness_clear_v1/
  signalized_lead_follow_001/comparisons/
```

Pass-by comparisons:

```text
astra/artifacts/static_left_tesla_overtake_distilled_100k_v1_comparison/
astra/artifacts/static_left_bus_overtake_distilled_100k_v1_comparison/
astra/artifacts/static_left_patrol_overtake_distilled_100k_v1_comparison/
astra/artifacts/static_left_pedestrian_overtake_distilled_100k_v1_comparison/
```

All four long comparison videos decode as 801 frames at 20 FPS. Critical
frames at 10, 20, 25, 30, and 38 seconds passed visual review for projection,
grounding, camera-specific visibility, and clipping.
