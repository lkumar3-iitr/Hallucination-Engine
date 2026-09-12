# Hallucination Engine Current State

Status date: 2026-09-08

Accepted renderer configuration: `astra_passby_v1_close_hybrid`

## What Works

HE can render a virtual Tesla Model 3 from the existing 4320-sprite bank into
a CARLA camera sequence without a physical adversary in the background video.
The current ASTRA path uses actor pose, camera pose, camera intrinsics, bank
metadata, and bank alpha masks. Runtime CARLA RGB, instance masks, and target
boxes are not renderer inputs.

The renderer now handles the failure that motivated this work: an ego vehicle
approaching, drawing alongside, and overtaking a static HE adversary. It keeps
the visible fragment while the actor spans the camera plane and removes it
after all valid rays leave the viewport. It does not use actor-center depth as
a whole-object visibility decision.

The 4320 production bank remains the general-view source. A small, opt-in,
camera-height-matched Cartesian bank supplies the difficult close side-pass
views. The ASTRA selector reconstructs a coarse bank-only visual hull and uses
its runtime projection to rank the available sprites. On selector-only tests
it is within 0.02 normalized silhouette IoU of exhaustive 4320-sprite search
on all 138 evaluated poses.

## Accepted Scenario Result

Scenario:
`static_left_tesla_ego_overtakes_200.resolved_v2.json`

The complete 201-frame HE replay has exact camera transforms relative to the
frozen CARLA reference and 100% visibility agreement. Across 61 visible frames:

| Metric | Result |
| --- | ---: |
| Mean mask IoU | 0.9429 |
| Mean mask Dice | 0.9658 |
| Mean bbox IoU | 0.9564 |
| Center x MAE | 0.516 px |
| Bottom y MAE | 1.672 px |
| Width MAE | 1.918 px |
| Height MAE | 2.607 px |

The side-pass is visually continuous through frames 45-59. Frame 60 remains a
small border-tail outlier. Frames 61-200 are correctly empty. Relative to the
previous native-only accepted replay, mask IoU improved from 0.9291 to 0.9429
and bottom-y MAE improved from 3.787 px to 1.672 px.

Evidence:
`astra/artifacts/static_left_tesla_overtake_targeted_close_v1_comparison/`

Of the 61 visible Tesla frames, 34 use `cartesian_close` and 27 use the 4320
`view_matrix` bank.

## Patrol Targeted-Hybrid Result

The height-matched Patrol close bank removes the upward-looking close-pass
selection seen with the older generic close bank. Across 201 frames and 62
jointly visible frames, visibility agreement is 1.0000, mean mask IoU is
0.9464, mean Dice is 0.9722, and mean bbox IoU is 0.9710. Center-x MAE is
0.427 px and bottom-y MAE is 1.823 px.

This supersedes the earlier current-compositor Patrol result of 0.9088 mask
IoU and 0.9398 bbox IoU.

Evidence:
`astra/artifacts/static_left_patrol_overtake_targeted_close_v1_comparison/`

Of the 62 visible Patrol frames, 34 use `cartesian_close` and 28 use the 4320
`view_matrix` bank.

## Pedestrian Targeted-Hybrid Result

The CARLA walker reference recorder is now fixed. Walker render state requires
two flushed synchronous settle ticks after each present/hidden calibration
transform; these ticks occur before frame 0 and do not alter recorded cadence.

The matched 201-frame pedestrian pair has 1.0000 visibility agreement across
52 visible frames, 0.8060 mean mask IoU, 0.8893 mean Dice, and 0.8325 mean bbox
IoU. Bottom-y MAE is 1.231 px. This supports continuous experimental pedestrian
rendering, but it remains below the accepted rigid-vehicle fidelity level.

Evidence:
`astra/artifacts/static_left_pedestrian_overtake_targeted_close_v1_comparison/`

## Selector Performance

Exact angular prefiltering and bit-packed silhouette IoU reduce Tesla native
selection from 125.0 ms to approximately 35-38 ms per call, about 3.4 times
faster. Recorded-pose checks have zero selected-key mismatches for Tesla,
Patrol, and pedestrian. A full optimized Tesla replay has zero mask-hash
mismatches across all 201 frames relative to the frozen accepted replay. See
`OPTIMIZATION_RESULTS.md` and the immutable hash record under `astra/frozen/`.

Independent 125-frame sweeps at native camera yaws 0, -60, and -90 degrees
have zero ASTRA dropouts and zero post-exit ghosts. Visible-frame mean IoUs are
0.9397, 0.9414, and 0.9452. Directly alongside frames 52-72 in the -90-degree
sweep average 0.9607 IoU.

## Long Closed-Loop Result

ASTRA also completed the 40-second, three-camera NEAT scenario
`signalized_lead_follow_001` through the generic closed-loop runner:

| Check | Result |
| --- | ---: |
| Logged frames | 801 / 801 |
| Front video frames | 801 at 20 fps |
| All-camera video frames | 801 at 20 fps |
| Front actor visibility | 801 / 801 frames |
| Right-camera actor visibility | 394 / 801 frames |
| Left-camera actor visibility | 0 / 801 frames |
| Minimum route bumper gap | 4.184 m |
| Minimum physical clearance | 4.161 m |
| Collision or overlap rows | 0 |

The actor remains continuous ahead, enters the right camera during the authored
right turn, and returns to the front view afterward. NEAT stops while the
traffic light is red, resumes after it changes green at 30 seconds, and finishes
the complete scenario. This run deliberately disables scene-depth occlusion for
ASTRA, and all per-camera `scene_depth_used` fields are zero.

Evidence:
`astra/artifacts/signalized_lead_follow_astra_final_v1/signalized_lead_follow_001/neat/`

The optimized close-hybrid renderer was subsequently rerun as a fresh matched
CARLA/HE pair with NEAT's native three-camera stack. Both conditions completed
all 801 frames. HE rendered the lead in the front camera for 801/801 frames,
the right camera for 400/801 frames during the turn, and the left camera for
0/801 frames as expected. Critical-frame review at frames 0, 100, 300, 400,
500, 580, 600, 650, 700, and 800 found continuous front/right rendering with
no obvious close-pass bending or upward tilt.

The synchronized 40.05-second comparison stacks CARLA and HE three-camera
mosaics and is available at:
`astra/artifacts/signalized_lead_follow_astra_optimized_pair_v1/comparison/signalized_lead_follow_001_neat_carla_vs_astra_all_cameras.mp4`

Because CARLA and HE are separate closed-loop conditions, their ego
trajectories may diverge over time; this video is a synchronized behavioral
comparison rather than a pixel-registered renderer metric.

## Bus Static-Overtake Result

The Fuso Rosa bus now has a correctly carved bank-only hull and a matched
authored static-overtake pair. Its native bank contains 5,760 views at
distances 10/15/20/25 m. Hull construction now chooses the nearest distance
available in each bank instead of assuming every bank contains 5 m captures.
For the visible side-pass, the bus uses a 252-view level, trajectory-matched
Cartesian-close bank and falls back to native ASTRA outside that coverage.

Across 201 frames, visibility agreement is 1.0000, mean bbox IoU is 0.9643,
and mean mask IoU is 0.9099. Center-x MAE is 0.500 px and width MAE is 1.194
px. The main pass is visually continuous; frames 69-71 remain border-tail
outliers. See `BUS_PASSBY_RESULTS.md` for the derivation and evidence paths.

## Renderer Architecture

1. The sprite bank supplies appearance, exact capture calibration, alpha masks,
   and physical bounding-box metadata.
2. A bank-only visual hull predicts the vehicle outline for a requested pose.
3. The selector ranks existing sprites against that predicted outline.
4. A virtual camera at the native camera origin is aimed along the separating
   direction to the closest point on the actor box. Selection and full
   unclipped placement occur in this stable camera.
5. A pure camera rotation maps the selected sprite into the native image.
   Inverse sampling performs viewport clipping and discards invalid rear rays.

`astra/compositor.py` adapts this path to the existing multi-actor result
contract. `astra/run_pair.py` injects it into the existing pair recorder for
one process. The generic closed-loop runner also exposes it explicitly through
`--he-renderer-version astra`. It remains opt-in rather than the repository-wide
default backend.

When an actor has a registered Cartesian-close bank and the runtime query lies
inside its calibrated coverage, the compositor uses that close sprite first.
It returns to ASTRA automatically outside the close bank. Bus hybrid frames are
recorded as `astra_passby_v1_close_hybrid`; native frames remain
`astra_passby_v1`.

## Version Decision

Use `astra_passby_v1_close_hybrid` with the asset-specific manifest as the
frozen renderer configuration for current Tesla and Patrol paper experiments.
The production 4320 banks remain unchanged and are still used outside targeted
close-bank coverage. No additional temporal smoothing is applied. In the
accepted scenario, selected bearing progresses monotonically with viewpoint;
the visible sequence has bank-distance transitions at frames 20 and 39 and an
elevation transition at frame 48. Adding history-dependent hysteresis now
would risk lag during the rapid close pass without evidence of a visual gain.

Use the explicit ASTRA backend for new Tesla, Patrol, and isolated bus
experiments. Do not replace or remove the legacy renderer yet: ASTRA
scene-depth occlusion and general multi-actor overlap still require validation. After those gates,
the generic runner default can be changed in a separate version decision.

Record native frames as `astra_passby_v1` and close-bank frames as
`astra_passby_v1_close_hybrid` in per-actor metadata. Keep the old renderer and
native-only manifest available as baselines.

## Remaining Limits

- RGB illumination and reflections differ because sprites preserve source
  capture appearance.
- The final few border pixels are less accurate than the main visible pass.
- Scene-depth occlusion is explicitly unsupported by the ASTRA adapter.
- Multi-actor overlap has not been validated. Patrol and isolated bus pass-by
  rendering are validated; pedestrian rendering remains experimental.
- The selector is approximately 3.4 times faster on the accepted Tesla native
  selections, but the complete renderer is not yet real-time.
- Contact/intersection with the virtual vehicle is outside the validated domain.

Multi-asset pass-by evaluation validates the Nissan Patrol bank above 0.91
mean visible IoU at all three tested camera yaws, with no dropouts and one
83-pixel border-tail ghost. Pedestrian rendering is continuous but remains
experimental at 0.80-0.89 mean visible IoU. The corrected bus hull removes the
previous near-plane dropouts and passes the authored static-overtake test; its
extreme synthetic sweep remains less accurate than the road scenario.

A targeted-hybrid Patrol authored overtake reaches 1.0000 visibility agreement,
0.9710 bbox IoU, and 0.9464 mask IoU across 62 visible frames. The pedestrian
authored pair now has valid matched evidence, but its 0.8060 mask IoU keeps it
experimental rather than at the accepted rigid-vehicle fidelity level.

These limits do not invalidate the static-adversary overtake result, but they
must remain explicit in paper claims. Current evidence supports simulator
rendering equivalence for the tested geometry and scenario, not general visual
or behavioral equivalence across all assets, cameras, and interactions.

## Reproduction

Run the test suite:

```powershell
python -m unittest discover -s astra -p test_*.py
```

Run the accepted scenario against an existing CARLA reference:

```powershell
python astra/run_pair.py --condition he --resolved important_curated_scenarios/resolved/static_left_tesla_ego_overtakes_200.resolved_v2.json --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --manifest astra/tesla_sidepass_asset_manifest_v1.json --reference-dir web/carla_reference/static_left_tesla_ego_overtakes_200 --output-dir astra/artifacts/my_final_replay --actor-id left_tesla --max-frames 201
```

Generated evidence stays under `astra/artifacts/` and is intentionally not
tracked by Git. See `SIDE_PASS_RESULTS.md` for detailed evidence paths.
