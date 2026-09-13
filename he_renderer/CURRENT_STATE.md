# HE Sprite Renderer Current State

The current calibrated backend is `he_calibrated_renderer_v2`; see
[MIGRATION_V2.md](MIGRATION_V2.md). Offline migration gates passed; live smoke
is pending simulator restart. The historical version-1 state below is retained
as baseline evidence, not the current v2 feature or performance specification.

## Version-1 Historical State

Status date: 2026-09-08

Accepted renderer: `he_sprite_renderer_v1`

## Supported Behavior

The renderer handles approach, directly-alongside motion, viewport clipping,
and exit during an ego overtake of a static virtual actor. It avoids using
actor-center depth as a whole-object visibility decision. The accepted hybrid
uses asset-specific close Cartesian banks where available and falls back to
the native 4320-view bank elsewhere.

Validated assets are Tesla Model 3, Nissan Patrol, Mitsubishi Fuso Rosa bus,
and pedestrian. Tesla, Patrol, and bus are accepted for current experiments.
Pedestrian support is functional but remains experimental because its matched
silhouette score is lower and animation is not represented by a static bank.

## Accepted Evidence

On the 201-frame static Tesla overtake, all 61 visible frames were detected:

| Metric | Result |
| --- | ---: |
| Mean mask IoU | 0.9429 |
| Mean mask Dice | 0.9658 |
| Mean bbox IoU | 0.9564 |

The Patrol overtake achieved 1.0 visibility agreement, 0.9464 mean mask IoU,
0.9722 Dice, and 0.9710 bbox IoU. The corrected pedestrian pair achieved 1.0
visibility agreement, 0.8060 mask IoU, 0.8893 Dice, and 0.8325 bbox IoU.

The exact optimized renderer preserves choices while reducing the measured
1280x720 native render from 116.1 ms to 32.5 ms. The complete optimized
201-frame Tesla replay produced masks identical to the accepted
pre-optimization replay by SHA-256 comparison.

An optional guarded distilled selector now reaches 120.7 FPS on the accepted
single-actor, single-camera native trajectory after warm-up. Its fresh matched
201-frame CARLA comparison retained 1.0 visibility agreement, 0.9423 mean mask
IoU, 0.9654 Dice, and 0.9562 bbox IoU. See `OPTIMIZATION_RESULTS.md` for the
teacher-table protocol, broad validation, and exact fallback behavior.

The 40.05-second `signalized_lead_follow_001` NEAT run completed 801 matched
frames in both CARLA and HE conditions. The front camera contributed 801
frames, the right camera 400 frames, and the left camera no visible actor
frames. The generated all-camera comparison passed visual review.

## Current Limits

- Sprite RGB retains source-capture illumination and reflections.
- Scene-depth occlusion is not implemented in this compositor.
- Multi-actor overlap has not received the same validation as single-actor passes.
- The selector is faster but not yet real-time optimized.
- Pedestrian articulation and animation are not modeled.
- Contact or intersection with the virtual actor is outside the validated domain.

Claims should remain limited to the tested CARLA geometry, assets, cameras,
and scenarios. The renderer establishes matched simulator rendering behavior;
it does not yet establish general real-world visual equivalence.
