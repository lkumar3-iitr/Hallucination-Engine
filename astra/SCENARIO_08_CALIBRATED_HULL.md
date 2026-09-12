# Scenario 8: Calibrated Hull Candidate

Date: 2026-09-11

Candidate identity: `astra_calibrated_hull_candidate_v1`.

Status: implemented and tested inside ASTRA; **not promoted to he_renderer**.
The earlier accepted ASTRA files and banks remain intact. This is a geometry
and continuity candidate, not a claim of complete photometric equivalence or
an optimized production replacement.

## What Caused the Regression

The accepted earlier bus pass used a lower, forward-facing camera. That result
did not validate NEAT's higher camera and its rotated left/right views.

The previous close path interpolated bounding boxes but selected a discrete
sprite image. Its production yaw guard also rejected close rendering for the
rotated cameras. Consequently, the front camera could use the new close bank
while the left camera fell back to native selection with large view changes.
For example, the earlier scenario log selected left-camera angles 106, 164,
and 92 degrees at frames 235, 240, and 245. A good silhouette IoU on a single
front frame did not establish rigidity or temporal quality.

## Candidate Geometry

1. Capture the whole bus from calibrated, level cameras. Aim horizontally at
   the midpoint of its angular bounding-box bounds, not simply its center.
2. Express camera position in actor coordinates. Bank selection is independent
   of the native camera's yaw, pitch, roll, resolution, and FOV.
3. Refine the existing bank-only visual hull using the new capture alpha masks.
4. Intersect native image rays with that hull. Project the resulting surface
   points through each neighboring capture's actual calibration and crop.
5. Interpolate the sampled premultiplied colors and alpha. Blend to the native
   baseline at every finite coverage boundary.

For target camera origin o and ray d in actor coordinates, the hull supplies
a surface point P = o + t d. Each source samples at
u = cx + fx * P_source.y / P_source.x and
v = cy - fy * P_source.z / P_source.x, with its recorded crop offset removed.
No target-box fitting or vertical anchor adjustment is applied in this path.

At an identical source/target camera origin, calibrated reprojection is an
exact camera rotation. Between origins, the bank-only hull remains a coarse
geometry approximation. This is not exact mesh recovery. An initial
single-side-plane approximation was rejected because it shifted the bus end
face and mirrors; its evidence is preserved as an intermediate experiment.

Runtime target RGB, target segmentation, target boxes, and CARLA depth are not
inputs to this close renderer. No inpainting is used. The generic closed-loop
runner retains its existing sensor setup; this change does not add scene-depth
or inter-actor occlusion support.

## New Bank

`artifacts/bus_aimed_close_v3/`

- 784 complete, non-clipped RGBA captures, approximately 159 MB of PNGs.
- Forward offsets: -12 through +12 m, every 0.5 m.
- Signed right offsets: -4.5, -3.5, +3.5, +4.5 m.
- Target-up levels: -0.3, -0.1726465702, +0.5749115236, +0.7 m.
- Capture size: 1536 x 1024; horizontal FOV: 150 degrees.
- CSV records actual actor/camera transforms, focal lengths, and crop offsets.
- Geometry-aware camera rejection and capture-edge checks are mandatory.

The existing bus native bank contains 5760 views, not 4320. It remains the
general-view source and initial visual-hull source. Native and close banks
outside ASTRA were not modified. Partial v1 and intermediate v2 banks remain
under artifacts; use v3 for this candidate.

## Frozen-Pose Geometry Tests

Independent CARLA captures use held-out positions, exact instance masks, and
clean backgrounds. NEAT tests contain 121 poses per camera, x=-18..12 m at
0.25 m spacing, camera height 2.3 m, lateral position 4.2 m, 400x300, FOV100.
The source grid is referenced to the bus bounding-box center, so these are
not exact capture-node replays.

Metrics use actual compositing alpha > 10/255, not RGB difference against the
background. Earlier RGB-difference-based reports are not strictly identical
metric protocols.

| Test | Visible frames | Mean mask IoU | Dropouts | Empty-GT frames with HE pixels |
| --- | ---: | ---: | ---: | ---: |
| NEAT front, yaw 0 | 82 | 0.9727 | 0 | 1 |
| NEAT left, yaw -60 | 98 | 0.9833 | 0 | 0 |
| NEAT right, yaw +60 | 0 | not applicable | 0 | 0 |
| Lower camera, 1280x720 FOV90 | 72 | 0.9606 | 0 | 0 |

The remaining front-camera outlier is frame 82: GT is empty and HE has 292
pixels. This is a real residual border-tail error, not a successful visibility
match. Do not omit it or classify the candidate as pixel-perfect. The right
camera is an empty-view check, not positive evidence of visible-bus quality on
that side. The new opposite-side bank was captured but a mirrored overtake
has not yet been evaluated.

Evidence directories:

- `artifacts/bus_neat_three_camera_reference_v1/`
- `artifacts/bus_calibrated_front_final_v1/`
- `artifacts/bus_calibrated_left_final_v1/`
- `artifacts/bus_calibrated_right_final_v1/`
- `artifacts/bus_calibrated_lower_final_v1/`

Each evaluation contains `comparison.mp4`, every reviewed frame, true-alpha
metrics in `rows.csv`, source metadata, and a summary.

## Full Scenario 8

Both conditions completed all 801 frames at 20 simulation Hz: 40 seconds plus
the initial sample. Weather: ClearNoon. NEAT: three native cameras. The same
resolved scenario, route trigger at 55 m, source-event start at 12 s, and
trigger-gated hidden pedestrian were used. No model/scenario tuning was made.

| Measurement | CARLA | HE candidate |
| --- | ---: | ---: |
| Trigger frame | 201 | 200 |
| Final logged route progress | 200.760 m | 202.058 m |
| Final speed | 6.945 m/s | 6.575 m/s |
| Minimum logged actor clearance | 1.138 m | 1.134 m |
| Geometric physical-overlap frames | 0 | 0 |

HE slowed and briefly stopped near the pedestrian before continuing. CARLA
had a different speed/braking history, including a later traffic-light stop.
Similar final progress does not establish behavioral equivalence. These are
independent closed loops, not pixel-aligned renderer pairs.

The source-selection audit covers all 2403 camera renders. Close capture
nodes and weights agree across all cameras. The main calibrated interval is
frames 216-380, with no mid-pass fallback. Entry blend is 205-215; exit blend
is 381-386. The later all-axis boundary guard changes none of the recorded
weights (maximum difference 0). The guard and a metadata-only projection label
correction were checked after the run; the image-generating core is unchanged.

Visual review included road frames 180-370 and the frozen-pose sequences.
At frame 240, CARLA and HE happen to have near-equal route progress, 63.363 m
and 63.353 m: the bus body is rigid and closely aligned in front/left views.
Source lighting/reflections and the missing cast shadow remain apparent. A
shadow-free sprite can look less grounded even with aligned tire geometry.

Primary video:
[40-second three-camera comparison](artifacts/scenario08_calibrated_comparison_v2/closed_loop_comparison.mp4)

Raw evidence:

- `artifacts/scenario08_calibrated_he_v1/`
- `artifacts/scenario08_calibrated_carla_v1/`
- `artifacts/scenario08_calibrated_comparison_v2/`
- `artifacts/scenario08_calibrated_he_review_v1/`
- `artifacts/scenario08_calibrated_carla_review_v1/`

The generic runner uses its existing `he_sprite_renderer_v1` CLI compatibility
alias. `CANDIDATE_RUN.json` and per-actor metadata explicitly identify the
ASTRA candidate; do not label this video as the accepted production renderer.

## Code and Verification

- `calibrated_close.py`: calibrated source sampling and continuous weights.
- `hull_rays.py`: CPU/CUDA hull intersections and close-mask refinement.
- `calibrated_bus_renderer.py`: native/close candidate composition.
- `calibrated_compositor.py`: bus-only adapter; other actors retain their
  current production backend with new caches redirected inside ASTRA.
- `capture_aimed_close.py`: isolated, resumable bank capture with QA.
- `run_calibrated_closed_loop.py`: process-local runner injection.
- `evaluate_calibrated_close.py`: frozen-pose, true-alpha validation.
- `audit_candidate_run.py`: camera consistency and recorded-weight audit.
- `make_candidate_comparison.py`: explicitly labeled closed-loop comparison.

All 19 ASTRA tests pass, including calibration identity, world-frame
invariance, empty-view handling, normalized interpolation, and bank-boundary
continuity. Added modules compile. Runtime logs and the consistency audit
preserve code/bank fingerprints.

This candidate uses PyTorch ray marching and is not optimized. The diagnostic
HE run recorded 1.976 end-to-end FPS, with cold work and concurrent geometry
evaluations on the same GPU. This is not a clean throughput benchmark, and
must not replace the accepted renderer's prior FPS measurements.

## Reproduce

From the repository root, in the `he_neat` environment:

```powershell
python -m unittest discover -s astra -p test_*.py
python -m astra.audit_candidate_run --run astra/artifacts/scenario08_calibrated_he_v1
```

Example frozen left-camera evaluation (use a new output directory):

```powershell
python -m astra.evaluate_calibrated_close --reference astra/artifacts/bus_neat_three_camera_reference_v1/yaw_m60 --close-bank astra/artifacts/bus_aimed_close_v3 --native-bank D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3 --output astra/artifacts/my_bus_left_review --hull --refine-hull
```

The exact full closed-loop arguments are stored in each `CANDIDATE_RUN.json`.
Use `run_calibrated_closed_loop.py` with those arguments and a new output root
inside ASTRA. CARLA must be idle before bank capture or scenario execution.

## Before Promotion

Review the new video with the user; retain the border-tail limitation. Validate
the mirrored pass and wider camera/pose coverage. Then profile and optimize
this candidate under repeatable, isolated load. Do not change the production
renderer, manifest, model behavior, or paper claims solely on these results.
