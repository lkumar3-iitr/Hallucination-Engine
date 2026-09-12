# Complete Side-Pass Experiment, 2026-09-08

Accepted renderer candidate: `astra_passby_v1`.

## Outcome

The ASTRA side-pass path renders through approach, alongside, and departure
without the old center/support-depth rejection. It uses no runtime GT masks
for selection, size, placement, or clipping. All changes remain in `astra/`.

Start with these videos:

- [Directly alongside, camera yaw -90](artifacts/validated_passby_yaw_-90/passby_comparison.mp4)
- [Oblique side camera, yaw -60](artifacts/validated_passby_yaw_-60/passby_comparison.mp4)
- [Original road pass-by, clean CARLA background](artifacts/passby_hull_size_v1/passby_comparison.mp4)
- [Full 201-frame static-adversary scenario](artifacts/scenario_static_overtake_full_v1_comparison/pair_comparison.mp4)

The complete sweeps use paired physical-target/target-absent CARLA recordings.
There is no inpainting. Each video contains every evaluated frame, including
empty views, clipped silhouettes, and both sides of the actor-center crossing.

## Authored Static-Adversary Scenario

The exact authored scenario
`important_curated_scenarios/resolved/static_left_tesla_ego_overtakes_200.resolved_v2.json`
was replayed end to end through `run_pair.py`: 201 frames at 20 fps. Inputs were
the frozen CARLA reference and its exact saved ego trajectory. Accepted output:
`artifacts/scenario_static_overtake_full_v1`.

The repository pair comparator reports:

| Metric | Result |
| --- | ---: |
| Frames | 201 |
| Both visible | 61 |
| Visibility agreement | 1.0000 |
| Mean bbox IoU | 0.9424 |
| Mean mask IoU | 0.9291 |
| Mean mask Dice | 0.9562 |
| Center x MAE | 1.041 px |
| Bottom y MAE | 3.787 px |
| Width MAE | 3.426 px |
| Height MAE | 4.344 px |

Pair validity is strong: maximum camera position/angle errors are zero, actor
position error is 3.81e-6 m, and geometric-view error is 3.12e-5 degrees.
Frames 45-58 have mask IoU 0.889-0.980 while the car grows and clips at the
left edge. Frame 59 scores 0.819. Frame 60 is the exit-tail outlier: CARLA has
only 38 pixels at the border, HE has 1099, and IoU is 0.034. Frames 61-200 are
correctly empty. This run therefore fixes the prior alongside disappearance,
while precise subpixel disappearance at the final border sliver remains open.

Evidence:

- `artifacts/scenario_static_overtake_full_v1/he_replay.mp4`
- `artifacts/scenario_static_overtake_full_v1_comparison/pair_comparison.mp4`
- `artifacts/scenario_static_overtake_full_v1_comparison/summary.json`
- `artifacts/scenario_static_overtake_full_v1_comparison/comparison.csv`
- `artifacts/scenario_static_overtake_full_v1_review/critical_frames_contact_sheet.png`

## Diagnosed Failures

1. `project_virtual_actor` rejects a whole actor when its center or selected
   support plane is near/behind the camera. This is not a valid whole-object
   visibility test when the vehicle extends across the camera plane.
2. Projected-box sprite selection requires all eight corners in front of the
   native camera. That condition fails during otherwise visible side passes.
3. The existing silhouette correction treats `anchor_x` as the horizontal
   alpha-box center, but the bank stores a bottom-contact anchor there. They
   differ. A controlled anchor-only correction substantially improved the
   preserved size path, but did not remove all box approximation error.
4. The multi-actor compositor also filters candidates by actor-center forward
   distance before rendering. The opt-in ASTRA compositor removes that gate.

On the new forward sweep the tested existing renderer stopped rendering at
frame 51 with 65345 GT pixels still visible. It missed frames 51-60 due to
support-plane/center-depth guards. On the -60-degree sweep it missed late
frames 64-73 while CARLA still showed the vehicle. ASTRA rendered all of these.
Baseline is the current local renderer invoked with `physical_bbox_silhouette`,
`oriented_2p5d_support`, and `projected_bbox_match`, not an old released binary.

## Fix

`passby_renderer.py` creates a virtual camera at the real camera's position,
aimed at the vehicle's physical bbox center. It selects and places the sprite
in this camera, where the vehicle remains in front throughout a non-contact
side pass. It then inverse-samples the sprite into real-camera pixels using
the exact camera-rotation mapping:

```text
p_virtual ~ K_virtual R_virtual^T R_native K_native^-1 p_native
```

The implementation includes the CARLA-to-optical-camera axis conversion.
The cameras share their origin, so this change of image coordinates is exact
for 3D points regardless of point depth. Sprite shape/appearance approximation
still comes from the finite bank, not from the rotation formula.

Only positive-depth source rays are sampled. This prevents mirrored ghost
images after departure. Inverse sampling into the native viewport avoids
allocating an enormous image when a projected corner approaches infinity.

Three box modes were compared:

- `existing`: unchanged production silhouette corrections in the virtual view.
- `anchor_fixed`: same width/height corrections with actual alpha-box anchors.
- `hull` (default): full projected bounds of the reconstructed bank-only hull.

The default gave the strongest full-image result. It does use the bank hull
for the box as well as selection; it is not a claim that the existing box path
was left numerically identical. A `box_provider` callback can supply an external
boxFinder in the virtual camera. It must return full, unclipped bounds.

## Complete-Sweep Validation

Accepted capture: `artifacts/complete_sweeps_v2`, 375 synchronized pose pairs:
125 frames each at camera yaws 0/-60/-90 degrees. Camera moves from x=-18.6 to
x=+18.6 m, at 0.3 m/frame, y=3.5 m, height 1.55 m above actor origin. Actor is
isolated at z=80 m. FOV 90 degrees, image 1280x720, 20 fps. Frame 62 is alongside.
The first capture version is superseded because the newly spawned vehicle's
first annotation had incomplete render initialization; v2 adds warm-up ticks.

| Camera | Total frames | GT-visible frames | Visible mean IoU | Dropouts | Ghosts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward, 0 deg | 125 | 61 | 0.93973 | 0 | 0 |
| Side, -60 deg | 125 | 73 | 0.94138 | 0 | 0 |
| Side, -90 deg | 125 | 43 | 0.94516 | 0 | 0 |

These are full native-frame silhouette IoUs, not size-normalized oracle scores.
GT enters only after rendering. Dropout means GT >20 pixels and HE zero pixels;
ghost means GT zero pixels and HE >20 pixels. No frame is skipped in evaluation.
For the -90-degree camera, alongside frames 52-72 average 0.96066 IoU, minimum
0.94543. Empty-frame IoU is explicitly defined as 1 and is not included in the
visible-only means above. Full per-frame results and reasons are under
`artifacts/validated_passby_yaw_{0,-60,-90}/`.

Existing recordings were also checked:

| Recording | GT-visible mean IoU | Late visible, frame >=40 |
| --- | ---: | ---: |
| Forward road pass-by | 0.92997 | 0.89931 |
| Yaw -60 | 0.94711 | 0.96419 |
| Yaw -80 / pitch -10 | 0.94054 | 0.96268 |

All three had zero dropouts, ghosts, and render errors by the above definitions.
The original forward exit frame 60 has only 38 GT pixels and 1099 predicted
pixels (IoU 0.0346). This residual edge/extent error is preserved in the scores;
the result is not pixel-perfect. New-sweep visible minimum IoUs are 0.5756,
0.2911, and 0.7637, with low scores concentrated around entry/exit slivers.

## Use

Offline replay of the original road sequence with its clean background:

```powershell
python astra/evaluate_passby.py --reference web/carla_reference/static_left_tesla_ego_overtakes_200 --background web/background_reference/static_left_tesla_ego_overtakes_200 --output astra/artifacts/my_passby --end 100
```

Direct API:

```python
from astra.passby_renderer import PassbyRenderer

renderer = PassbyRenderer(bank_path, "astra/artifacts/cache")
bgr, alpha, metadata = renderer.render(background_bgr, actor_tf, camera_tf, 90.0)
```

`compositor.py` implements the existing recorder's compositor result contract.
`run_pair.py` injects it into the recorder module for that Python process only;
the production files and their default backend remain unchanged. This launcher
accepts the existing recorder arguments, requires `--condition he`, and confines
its output to a new directory under `astra/`. It bypasses the Cartesian close
bank and the old actor-center candidate gate. Backend identity is written into
setup metadata and each actor's rendering metadata.

The launcher was exercised against live CARLA using the original road
reference. `artifacts/recorder_integration_v2` contains a completed 80-frame
HE replay, 80 reconstructed-alpha masks, and a separately recorded clean
background. Camera transforms match the reference exactly. Its 61 GT-visible
frames average 0.92915 mask IoU, with zero dropouts and ghosts. The recorder
uses nonzero reconstructed alpha, whereas sweep evaluation uses alpha >10/255.
The preceding v1 integration attempt failed asset-root validation and is not
an accepted run.

```powershell
python astra/run_pair.py --condition he --resolved important_curated_scenarios/resolved/static_left_tesla_ego_overtakes_200.resolved_v2.json --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --reference-dir web/carla_reference/static_left_tesla_ego_overtakes_200 --output-dir astra/artifacts/my_recorder_replay --actor-id left_tesla --max-frames 80
```

## Boundaries

This fixes the tested side-pass geometry/visibility failure, not every rendering
problem. Lighting/reflections, finite-bank perspective detail, temporal view
changes, contact/intersection with the virtual vehicle, and precise entry/exit
silhouettes remain imperfect. Runtime has not been optimized for real-time use.
The recorder adapter explicitly rejects scene-depth occlusion input rather
than silently ignoring it. Multi-actor overlap and other assets are not validated.
The production renderer is not globally switched to ASTRA.

## Verification

Twelve unit tests pass, including exact camera-rotation agreement with 3D
projection, actor-center crossing, visible fragments across the native camera
plane, prevention of behind-camera ghosts, and rejection of translated-camera
homographies. All ASTRA Python files compile. `verify_passby.py` checks the
375-frame sweeps, complete-frame video decoding, masks, score claims, and the
80-frame recorder integration; its report with current code hashes is
`artifacts/passby_verification.json`. Alongside and clipped RGB frames were
visually inspected. CARLA was left asynchronous with no vehicle/sensor actors
remaining from these experiments.
