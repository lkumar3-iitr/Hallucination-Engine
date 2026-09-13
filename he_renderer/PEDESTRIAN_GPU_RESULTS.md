# Pedestrian GPU Optimization Checkpoint

## Live Scenario08 Addendum

`artifacts/scenario08_pedestrian_gpu_live_v1` completed 801 frames in ClearNoon
using NEAT and three synchronized depth cameras. Explicit launcher flag:
`--he-pedestrian-gpu` with `he_calibrated_renderer_v2`. The default remains off.
Source archive and receipt record this opt-in choice.

- Final route progress: 155.29 m; final speed: 6.60 m/s; trigger frame: 197.
- Zero logged geometric overlap frames (not a physical collision-sensor claim).
- Wall time: 54.99 s; end-to-end throughput: 14.57 FPS.
- Logged HE mean: 22.34 ms per three-camera set, P95 31.50 ms.
- Model inference mean: 21.79 ms. Timing excludes the first 20 frames for
  stage statistics, but later first-use costs remain (HE maximum 530.08 ms).
- 2,403 camera calls; all 1,812 active pedestrian calls carry the candidate
  identity and enabled scene-depth metadata. All 2,403 bus calls report depth.

Comparison: `artifacts/scenario08_pedestrian_gpu_comparison_v1/comparison.mp4`,
801 decoded frames, against the earlier physical run in
`artifacts/evaluation_v2_scenario08/carla`. Independent closed-loop trajectories
at equal time, not matched-pose image metrics. The HE run continued after the
encounter, whereas the earlier default HE run stopped; neither outcome is
discarded and the cause of behavioral variation is not established. This is
not a controlled algorithm-speedup estimate relative to the stopped run.

Frames 240 and 400 were visually inspected. Road-contact appearance of the bus
still looks elevated in these views; this must not be declared solved from
the successful run or attributed to pedestrian acceleration without matched
inputs. Sparse visual inspection is not a full fidelity gate. Default promotion
remains pending visual review and investigation of that placement appearance.
No bank, default, paper table, commit or push changed in this live pass.

## Continuous Review Addendum

Completed offline review in `artifacts/pedestrian_gpu_continuous_v1`:
`pedestrian_comparison.mp4` and `bus_pedestrian_comparison.mp4`, each 201
decoded frames at the recorded frame rate. Default and candidate RGB are
exactly equal on all 402 frames before video encoding. The pedestrian is
visible at frames 0-51, then absent in both versions; the synthetic bus is
visible at frames 0-111. Selected approach/exit and close-bus frames were
visually inspected, not every video frame manually.

These use the existing clean Tesla-overtake recording's camera trajectory,
resized to 640x360 at its original FOV, with synthetic pedestrian substitution
and a bus placed 12 m farther along the actor heading. No simulator or target
ground truth is used. Scene depth is absent in these videos; the separate
512-case numerical gate covers depth. This is not Scenario08 replay.
The synthetic bus appears elevated above the road in inspected frames in
both versions: matching outputs does not validate this placement setup.
Do not use these videos as physical fidelity evidence. Production defaults
and the paper baseline remain unchanged. A recorded/live Scenario08 comparison
with its actual actor placement and synchronized depth remains pending.

Reproduce with `he_renderer/evaluation/review_pedestrian_gpu.py`, specifying
`--asset-root`, `--background web/background_reference/static_left_tesla_ego_overtakes_200`
and a new `--output` directory. Per-frame metadata and summary are retained
alongside the videos. No CARLA restart was needed for this offline review.

Date: 2026-09-12. Opt-in `pedestrian_gpu_candidate_v1` inside the calibrated
compositor. Default pedestrian rendering, banks and paper baseline unchanged.
CARLA remained closed; no model, live replay, commit or push performed.

## Implementation

Pedestrian already received warmups before this work. Runtime acceleration now
reuses vehicle GPU projection, integer rasterization, scoring and native warp.
The original pedestrian exact/distilled selection policy is retained: accepted
distilled choices bypass exact GPU selection, and rejected ones use the same
bearing restriction, scores and tie handling. No confidence threshold changed.
Exact bounded selection sharing is allowed only for identical camera queries.

Native depth traversal, ray rotation and camera-forward hit-depth calculation
stay on GPU. Only the depth image returns to CPU, not all 3-D hit points.
The immutable hull volume and eight-entry camera-ray LRU stay resident.
Native textures use the existing bounded 256 MiB cache. CPU compositing and
occlusion remain; this is not an entirely GPU-resident pipeline. These cache
budgets do not bound total process VRAM. Recreate engines after changing banks,
calibration or hulls. Instances remain non-thread-safe.

The targeted Cartesian pedestrian close renderer is unchanged, including its
older nearest-depth approximation. No calibrated pedestrian close bank was
enabled, created or reduced. Default constructor uses `pedestrian_gpu=False`.
Candidate actor metadata adds `performance_backend=pedestrian_gpu_candidate_v1`.

Before editing, source files were saved in
`artifacts/pedestrian_pre_gpu_reference.zip` (selected files only). The earlier
compositor FPS archives retain the broader pre-change source snapshot; neither
includes external banks or runtime dependencies.

## Validation

`artifacts/pedestrian_gpu_final_quality_v1` compares the opt-in compositor
against the unchanged default in 512 cases: four adapter camera profiles,
four offsets, both sides, actor yaws 0/35 degrees, and depth absent, foreground
strip, full foreground, or invalid (NaN, negative and infinite) values.

- Final RGB exact in all cases.
- Public placement, selected sprite and occlusion metadata match within
  1e-9 numerical tolerance, excluding the added performance identity.
- 126 native alpha comparisons: maximum error 5.96e-8, zero threshold-mask
  changes at 10/255. Cartesian close code is not replaced.
- 37 unit tests pass, including preservation of accepted distilled choices
  and routing rejected choices through exact selection.

These are finite reference-regression tests, not ground-truth physical
accuracy or proof for all poses. Raw depth-image error is not independently
bounded by the metadata check. No new video or live model evaluation was
performed. Existing HE-to-HE ordering limitations remain.

## Timing

Two isolated serial sweeps, `compositor_pedestrian_gpu_fps_v1/v2`, use the same
production-compositor timing boundary as PEDESTRIAN_FPS_RESULTS.md. CARLA is
closed, CPU input/output, synthetic aligned depth, ten warmups and 100 samples
per workload per run, shuffled workload order. Query caches clear per set;
textures stay warm. Includes metadata and stacking, excludes file logging,
video I/O, simulator, model inference and cold initialization.

Pooled camera sets/s / P95 set latency (ms), 800 samples per cell:

| Profile | Pedestrian | Bus + pedestrian | Bus + Tesla + Patrol + pedestrian |
| --- | ---: | ---: | ---: |
| TCP | 194.41 / 7.55 | 94.41 / 12.02 | 53.38 / 20.46 |
| NEAT | 100.06 / 10.71 | 61.81 / 18.31 | 32.88 / 32.73 |
| CIL++ | 126.65 / 8.57 | 72.12 / 15.82 | 36.81 / 29.72 |
| AIM-MT | 250.91 / 4.46 | 152.03 / 8.19 | 75.94 / 14.73 |

Each camera updates at the set rate. For NEAT/CIL++, aggregate images/s is
three times that rate. All 12,800 timed sets are RGB-repeatable. This is a
same-implementation repeatability test; the independent reference gate above
is single-actor. Mixed-asset physical occlusion correctness is not established.

Groups use successive longitudinal positions 0,10,... m, lateral -4.2 m,
and ego offsets 30/12/6/0 m. Some views are empty; changing groups changes the
pedestrian pose. Therefore columns do not isolate incremental actor cost.
NEAT's pooled mean barely exceeds 100 sets/s; its P95 exceeds 10 ms, so this
is not a guaranteed 100 Hz deadline. Four-asset workloads do not reach 100 Hz.

A separate block-paired diagnostic (`pedestrian_gpu_resident_depth_v1`),
alternating reference/candidate block order, yielded pooled pedestrian-only
reference -> candidate rates: TCP 67.83 -> 184.26, NEAT 20.56 -> 89.25,
CIL++ 27.01 -> 117.55, AIM-MT 65.14 -> 215.82 sets/s. Do not merge this
protocol with isolated timing or attribute all historical differences to this
edit. The unchanged TCP 6 m close control varied between blocks; no speedup
is attributed to that path.

The first prototype downloaded hit points and profiled at about 12 ms for
one AIM-MT alongside image, with about 7 ms in native surface-depth work.
The resident-depth prototype profiled near 6 ms total with about 1.4 ms in
surface-depth work. These are single-pose instrumented diagnostics, not an
exhaustive stage decomposition; GPU host returns include waiting time.
Retained evidence: `pedestrian_gpu_gate_v1`, `pedestrian_gpu_profile_v1`, and
`pedestrian_gpu_resident_depth_v1`.

## Reproduction and Next Gate

```powershell
conda run --no-capture-output -n he_neat python he_renderer/evaluation/validate_pedestrian_gpu.py --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --output he_renderer/artifacts/my_pedestrian_gpu_gate --quality-only
conda run --no-capture-output -n he_neat python he_renderer/evaluation/benchmark_compositor.py --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --output he_renderer/artifacts/my_pedestrian_gpu_fps --repeats 100 --pedestrian-gpu
```

For direct use, explicitly construct
`HECalibratedCompositor(pedestrian_gpu=True)`. Live launchers do not yet enable
this candidate. Review continuous pedestrian pass-by and a mixed bus/pedestrian
replay before default promotion. Keep the current paper table as the accepted
baseline until those gates complete. No CARLA restart is required for further
offline work; live comparison will require it later.
