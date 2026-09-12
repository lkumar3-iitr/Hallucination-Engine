# ASTRA Multi-Asset Pass-By Results

Status date: 2026-09-08

## Protocol

Each asset was evaluated in the same 125-pose pass-by sweep at native camera
yaws 0, -60, and -90 degrees. Every pose contains a physical CARLA render and
instance mask plus a matched clean background frame with the actor removed.
ASTRA receives only the clean frame, actor and camera transforms, intrinsics,
and the selected asset bank. CARLA masks are read only after rendering.

The new evaluation covers 375 poses per asset and includes fully visible,
viewport-clipped, camera-plane, empty, and post-exit frames.

## Results

| Asset | Camera yaw | Visible frames | Mean visible IoU | Minimum IoU | Dropouts | Ghosts | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nissan Patrol | 0 | 63 | 0.9155 | 0.4379 | 0 | 0 | 0 |
| Nissan Patrol | -60 | 75 | 0.9302 | 0.7275 | 0 | 1 | 0 |
| Nissan Patrol | -90 | 47 | 0.9127 | 0.2784 | 0 | 0 | 0 |
| Fuso Rosa bus | 0 | 72 | 0.5369 | 0.0000 | 16 | 0 | 18 |
| Fuso Rosa bus | -60 | 81 | 0.5485 | 0.0000 | 18 | 3 | 18 |
| Fuso Rosa bus | -90 | 66 | 0.4888 | 0.0000 | 18 | 6 | 18 |
| Pedestrian 0001 | 0 | 52 | 0.7964 | 0.3740 | 0 | 0 | 0 |
| Pedestrian 0001 | -60 | 51 | 0.7971 | 0.0000 | 0 | 0 | 0 |
| Pedestrian 0001 | -90 | 25 | 0.8850 | 0.7870 | 0 | 0 | 0 |

The Patrol ghost is one frame containing 83 predicted border pixels after the
CARLA mask becomes empty. The pedestrian zero-IoU frame contains only 20 CARLA
pixels and is below the dropout threshold.

## Interpretation

The Patrol bank transfers successfully to ASTRA. All three sweeps are
continuous, error-free, and above 0.91 mean visible IoU. Its one border-tail
ghost is small but should be covered by the planned exit-tail refinement.

The pedestrian is continuous and error-free, but its mean IoU at yaw 0 and
-60 is about 0.80. Thin articulated contours make a fixed coarse visual hull a
weaker selector than it is for rigid vehicles. This result is promising but is
not yet at the accepted Tesla/Patrol geometric-equivalence level.

The first bus sweep was invalidated by a hull-construction assumption: the
builder requested 5 m captures, while the bus bank starts at 10 m. Hull cache
version 2 now uses the nearest distance present in each bank and aims the
virtual camera along a closest-box separating direction. The corrected extreme
sweeps have zero dropouts and errors, with visible mean IoUs of 0.7071, 0.7667,
and 0.7597. The more representative authored static-overtake pair reaches
1.0000 visibility agreement, 0.9652 bbox IoU, and 0.8843 mask IoU. See
`BUS_PASSBY_RESULTS.md`.

## Evidence

All generated evidence is under:

`astra/artifacts/passby_all_assets_v1/`

Each asset contains `reference/yaw_*` CARLA videos, masks, clean backgrounds,
and transforms, plus `astra/yaw_*` comparison videos, CSV rows, and summaries.
Review sheets isolate the Patrol border tail, bus near-plane failure, and
pedestrian close pass.

## Decision Gate

- Accept Tesla and Nissan Patrol for current rigid-passenger-vehicle tests.
- Keep pedestrian support experimental until its silhouette selector improves.
- Accept isolated bus static-overtake rendering with an explicit border-tail
  limitation; do not extend that claim to contact/intersection.
- Preserve the corrected hull before performance and depth-occlusion work.

## Current-Compositor Authored Pairs

The height-matched targeted Patrol bank was tested on the matched static
overtake using the accepted 1280x720, FOV-90 camera. Across 201 frames and 62
jointly visible frames, visibility agreement is 1.0000, mean bbox IoU is
0.9710, mean mask IoU is 0.9464, and mean Dice is 0.9722. The complete pass is
continuous with no dropout or post-exit ghost. This supersedes the earlier
generic-close-bank result of 0.9088 mask IoU. The targeted bank supplies 34 of
the 62 visible frames; the production 4320 bank supplies the other 28.

Evidence:
`artifacts/static_left_patrol_overtake_targeted_close_v1_comparison/`

The equivalent Tesla targeted hybrid reaches 1.0000 visibility agreement,
0.9564 mean bbox IoU, 0.9429 mean mask IoU, and 0.9658 mean Dice across 61
jointly visible frames. Its opt-in configuration is
`tesla_sidepass_asset_manifest_v1.json`. The targeted bank supplies 34 of the
61 visible frames; the production 4320 bank supplies the other 27.

Evidence:
`artifacts/static_left_tesla_overtake_targeted_close_v1_comparison/`

The analogous pedestrian authored pair is now valid. The recorder fix adds two
flushed synchronous settle ticks after each walker calibration transform,
before recorded frame 0. Across 201 frames and 52 jointly visible frames,
visibility agreement is 1.0000, mean bbox IoU is 0.8325, mean mask IoU is
0.8060, and mean Dice is 0.8893. Bottom-y MAE is 1.231 px. The result confirms
continuous rendering but remains experimental because articulated silhouette
selection is weaker than rigid-vehicle selection.

Evidence:
`artifacts/static_left_pedestrian_overtake_targeted_close_v1_comparison/`

Targeted banks are nevertheless complete for both pedestrian sides: 102 views
per side over the observable grid, forward 2.25-10.5 m and lateral 3.0/3.5/4.0
m. The narrower 0.25-2.0 m corner samples were excluded because the rendered
body is outside the capture frustum even where its coarse physical box barely
intersects it. The matched replay uses `pedestrian_sidepass_asset_manifest_v1.json`.
