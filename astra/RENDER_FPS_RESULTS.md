# Renderer-Core FPS Test

## Latest Combined Rerun

`artifacts/render_fps_latest_v2/summary.json`: latest exact native optimizations,
fused traversal, with/without guarded lookup and exact cross-camera selection
cache. Three measured six-pose sweeps after one warmup sweep (18 samples per
configuration); no profiler. Same bus-only setup/exclusions below, fixed
backend order. Repeats improve timing stability but do not expand pose coverage.

| Actors | Single-camera FPS, fused exact | Single-camera FPS, combined | Combined three-camera sets/s | Combined aggregate images/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 22.11 | 22.02 | 11.59 | 34.78 |
| 2 | 12.74 | 13.70 | 5.06 | 15.18 |
| 3 | 6.38 | 6.95 | 3.58 | 10.74 |

For the combined candidate, single-camera median/p95 latency in ms:
one actor 45.0/76.8; two actors 65.5/142.4; three actors 138.7/219.7.
These are diagnostics from 18 observations, not reliable tail estimates.
FPS uses total frames divided by summed latency, not reciprocal median.
Aggregate images/s does not equal the update rate of each camera: with three
cameras, each camera's update rate is the camera-set rate.

Current measured single-camera throughput remains far below 100 FPS on this
mixed-distance load. Lookup provides negligible single-actor single-camera
benefit here; multi-camera exact reuse remains useful. Neither fused nor lookup
was promoted to runtime defaults by this benchmark. No quality changes made.

Reproduce with a new output directory:
`python -m astra.benchmark_render_fps --lookup --repeats 3 --output astra/artifacts/NEW_RUN`.

Date: 2026-09-12. Script: `benchmark_render_fps.py`.
Raw evidence: `artifacts/render_fps_v1/summary.json`.

Each image is 400x300 FOV100. One or three cameras (yaw 0/-60/+60).
One to three bus instances share the same bank/hull. Positions change across
six longitudinal offsets: 30,15,6,0,-6,-12 m. Actor instances are 3 m apart;
this artificial arrangement is a load test, not validated scenario geometry.

Timed work includes pose creation/order, native selection, projection,
traversal, source sampling, image composition and synthetic scene-depth
occlusion. Excludes CARLA, model inference, disk/video I/O, initialization,
production adapter metadata/logging and RGB/BGR boundary conversions. This
is complete renderer-core work, not the complete deployment adapter pipeline.
No per-pixel HE-to-HE z-buffer is implemented.

One sweep warms each configuration, followed by one timed six-frame sweep;
CUDA is synchronized around each complete camera set. Intersection cache is
cleared at the start of each set. Both backends retain within-render/depth
reuse. Current backend runs first, so shared source caches may favor the
fused candidate; this is preliminary diagnostic throughput, not paper-grade
benchmarking. Six samples are insufficient for reliable tail percentiles.

| Cameras | Actors | Current sets/s | Fused candidate sets/s |
| --- | ---: | ---: | ---: |
| 1 | 1 | 10.30 | 19.91 |
| 1 | 2 | 5.53 | 11.29 |
| 1 | 3 | 3.18 | 5.86 |
| 3 | 1 | 3.26 | 6.51 |
| 3 | 2 | 1.56 | 3.14 |
| 3 | 3 | 1.04 | 2.10 |

For three cameras, multiply sets/s by three to obtain images/s, not frame
sets/s. For example, fused one-actor three-camera throughput is 19.53 images/s
but 6.51 complete sets/s. Throughput is frame count divided by summed latency,
not the reciprocal of median latency.

The fused traversal is activated only inside this test and restored afterward.
It is not the default runtime backend. Banks/production remain unchanged.

Native fallback still dominates mixed-distance rendering. This explains why
millisecond traversal does not translate to hundreds of complete images/s.
Next: native projection/selection acceleration, followed by longer randomized
paired measurements on real replay trajectories and mixed asset banks.
