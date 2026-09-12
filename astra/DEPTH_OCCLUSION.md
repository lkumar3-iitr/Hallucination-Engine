# ASTRA Scene-Depth Occlusion

Date: 2026-09-12. Experimental, opt-in; production files unchanged.
Backup: `frozen/pre_depth_20260912.zip`.

## Contract

Provide an aligned HxW camera-forward depth image in metres with the same
pose, intrinsics and timestamp as the RGB frame. Not Euclidean ray range.
ASTRA validates shape/type; the producer must guarantee calibration/time.
The existing CARLA runner supplies synchronized decoded metric depth.

For calibrated assets, project each bank-hull first-hit point into the native
camera and compare its forward depth with scene depth. A pixel is hidden when
scene_depth + 0.05 m < hull_depth. Restore the original compositor background
at that pixel and zero its alpha. Other pixels retain their original RGB.
The tolerance is a numerical/geometry margin, not a calibrated accuracy claim.

Invalid scene depths (nonpositive or nonfinite) and missing hull surface
depth leave appearance unchanged and are counted in metadata. This avoids
silently erasing silhouette regions unsupported by the approximate hull,
but those regions cannot be claimed to have correct occlusion.

The retained baseline native path uses its bank hull. Its legacy Cartesian
close path uses the pre-existing nearest-depth approximation, explicitly
labeled `legacy_nearest_depth_approximation`; it is not per-pixel surface
depth. This matters for the retained pedestrian backend.

This handles real scene geometry occluding HE. It does not implement a shared
per-pixel HE-to-HE z-buffer, cast shadows, or physical collision response.
Virtual actors are absent from the physical scene depth. Ground truth target
masks are not inputs. No depth supplied means the previous rendering path.

## Run

Add `--candidate-scene-depth` to `astra/run_calibrated_closed_loop.py`.
The process-local runner adapter removes the legacy depth-withholding gate
using a checked AST transformation, and logs depth use from actor metadata
instead of the legacy renderer-version check. It fails if either gate changes;
it does not edit the common runner on disk. The receipt records depth use.

## Verification

- All 26 ASTRA tests pass, including foreground/background depth, invalid
  depth, unknown hull depth, shape rejection and rotated-camera forward depth.
- `python -m astra.smoke_scene_depth` renders an actual calibrated bus:
  a controlled foreground strip hides 11,174 of 37,060 visible pixels.
  Farther depth leaves RGB and alpha exactly unchanged. The comparison image
  was inspected: `artifacts/depth_smoke_v1/comparison.png`.
- NEAT live wiring smoke completed 41 samples with three synchronized cameras:
  `artifacts/scenario08_depth_wiring_smoke_v1`. This short approach run does
  not exercise the pedestrian or establish real-obstacle occlusion accuracy.
  This initial run has stale zero CSV depth-use flags; its renderer JSONL
  correctly records enabled depth. The logging correction is tested separately
  in `artifacts/scenario08_depth_wiring_smoke_v2` (three samples).

Full occlusion encounter, legacy pedestrian branch, and 40-second visual
regression acceptance remain pending. Do not label this benchmark-ready yet.
The first implementation repeats hull intersection for depth; optimize by
reusing hit points after validating correctness. No throughput improvement
is claimed for this feature.
