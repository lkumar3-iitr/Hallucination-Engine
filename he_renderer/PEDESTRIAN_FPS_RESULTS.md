# Pedestrian-Inclusive Offline Throughput

Later opt-in optimization: see [Pedestrian GPU checkpoint](PEDESTRIAN_GPU_RESULTS.md).
The measurements below remain the unchanged default-backend baseline.

Date: 2026-09-12. Production `he_calibrated_renderer_v2`, unchanged renderer.
CARLA closed; NVIDIA RTX PRO 4000 Blackwell under Windows, display active.

## Scope

CPU RGB, aligned synthetic depth and poses through HECalibratedCompositor
to stacked CPU camera images. Includes actor sorting, RGB/BGR conversion,
placement/visibility metadata and depth. Excludes cold initialization, video
I/O, file logging, simulator and model inference. This is broader than the
earlier renderer-core tables; do not interpret their difference as a regression
or a controlled measurement of compositor overhead.

Camera intrinsics/extrinsics are parsed from the TCP, NEAT, CIL++ and AIM-MT
adapters without loading model weights. Four ego offsets (30, 12, 6, 0 m),
ten warmups per workload, 100 samples per workload in each of two serial runs.
Workload order is shuffled with seeds 92184 and 92185. Synchronization bounds
timing. Exact vehicle selection and hull-hit caches are cleared per set;
within-set sharing and warm source/calibration caches remain. No temporal
approximation is added. CPU foreground depth is 1000 m with a 0.5 m strip.

Group actors occupy longitudinal positions 0, 10, ... m, lateral -4.2 m,
zero yaw and z. Physical dimensions come from asset metadata. These are
synthetic cost tests, not a replay of Scenario08, realistic traffic or a
simultaneously visible-actor guarantee. No HE-to-HE z-buffer is implemented.

## Results

Cells are pooled camera sets/s / P95 set latency in ms, 800 samples per cell.
FPS is total sets divided by total time, not arithmetic mean FPS. Each camera
updates at the set rate; NEAT/CIL++ aggregate images/s are three times that.

| Profile | Bus | Pedestrian | Bus + pedestrian | Bus + Tesla + Patrol + pedestrian |
| --- | ---: | ---: | ---: | ---: |
| TCP | 197.68 / 6.96 | 62.54 / 25.01 | 35.22 / 33.78 | 28.05 / 43.02 |
| NEAT | 118.86 / 11.53 | 20.66 / 53.86 | 18.75 / 74.81 | 12.78 / 90.71 |
| CIL++ | 134.02 / 9.37 | 26.53 / 47.05 | 24.40 / 63.07 | 15.58 / 79.68 |
| AIM-MT | 291.78 / 5.62 | 60.98 / 19.71 | 61.46 / 21.34 | 38.08 / 32.37 |

Both run rates, in the same column order (ranges are not confidence intervals):

| Profile | Bus | Pedestrian | Bus + pedestrian | All four |
| --- | ---: | ---: | ---: | ---: |
| TCP | 192.72-202.91 | 61.79-63.31 | 34.40-36.08 | 27.30-28.84 |
| NEAT | 118.78-118.95 | 20.27-21.05 | 18.74-18.76 | 12.72-12.84 |
| CIL++ | 132.30-135.79 | 26.12-26.96 | 24.18-24.62 | 15.40-15.77 |
| AIM-MT | 287.49-296.21 | 60.85-61.12 | 58.72-64.46 | 37.06-39.14 |

## Interpretation and Gates

Pedestrian still uses `legacy_pedestrian`, with its existing guarded distilled
selector where available and exact fallback otherwise. The targeted Cartesian
close bank and its older depth approximation remain unchanged. Vehicle GPU
throughput must not be attributed to this backend.

TCP pedestrian-only at offset 6 m uses Cartesian close rendering (about 608
FPS in both runs). Most other pedestrian views use native fallback. At offset
0 m the pedestrian is outside the single TCP/AIM-MT viewport. The multi-camera
sets include empty views. Per-camera actor metadata is retained for every
workload. Do not label all these samples native, close, or fully visible.

Adding an actor also changes which longitudinal position the pedestrian
occupies, so columns are not a controlled incremental per-actor cost estimate.
For example AIM-MT bus+pedestrian can beat pedestrian-only in the pooled
workload. Per-pose timing varies; both runs are retained, not cherry-picked.

All 12,800 timed camera sets match their untimed same-implementation RGB
reference exactly. This checks repeatability only, not independent rendering
fidelity or correctness of depth/occlusion. There are 64 unique workload
configurations repeated twice, not 12,800 independent scenarios. The production
unit suite passes 35 tests, including two new benchmark-contract tests.

The next performance target is pedestrian native selection/depth and its
CPU/GPU transfers. Preserve its accepted appearance and use separate visual
and depth equivalence gates before adopting vehicle-style GPU changes.
No rendering algorithm, bank, default, commit or push changed in this pass.

## Reproduction

```powershell
conda run --no-capture-output -n he_neat python he_renderer/evaluation/benchmark_compositor.py --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --output he_renderer/artifacts/my_pedestrian_fps --repeats 100 --seed 92184
```

Evidence: `artifacts/compositor_pedestrian_fps_v1` and
`artifacts/compositor_pedestrian_fps_v2`. Each contains raw samples, camera
source hashes, actor/path metadata, environment snapshot and source archive.
Archives exclude external banks, runtime dependencies and generated caches.
