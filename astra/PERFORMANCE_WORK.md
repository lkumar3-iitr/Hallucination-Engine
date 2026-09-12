# Calibrated Renderer Performance Work

Start with [Renderer Logic and Optimization Map](RENDERER_LOGIC_AND_OPTIMIZATIONS.md)
for the unoptimized algorithm, inputs/equations, optimization status and limits.

## Covered-Polygon Elimination

Added an integral-image coverage test after collapsed pixel runs. Residual
polygons whose complete inclusive bounding rectangle is already painted are
skipped: they cannot add pixels to the union. Remaining polygons retain the
same deduplication and OpenCV fill. No geometry/selection approximation.

Sampled warm profiled native rendering: 46.76-48.59 ms versus 58.10-58.36 ms.
RGB/alpha match in three far views; all 72 bus mask/box checks against the
original frozen selector pass exactly. All 29 tests pass. Close geometry,
banks and production files are untouched. Full combined FPS, other assets
and full replay regression have not been rerun for this edit.

Evidence: `artifacts/native_covered_polygon_v1`,
`artifacts/native_covered_polygon_validation_v1`; source backup:
`frozen/pre_covered_polygon_20260912.zip`.

## Ablation and Residual Polygon Deduplication

Deferred temporal coherence as requested. Added `--ablation` to the renderer
core benchmark, profiling fused traversal with (1) uncached exact selector,
(2) exact cross-camera cache, (3) guarded lookup plus cache. On the six-pose
diagnostic, three-camera/three-bus throughput was 2.2, 3.3, 3.5 sets/s
respectively. Exact sharing provides most of this workload's benefit; lookup
coverage is limited. These profiled, fixed-order tests are not paper-grade
throughput measurements. Evidence: `artifacts/render_profile_ablation_v1`.

Combined-path profile: exact selection fallback 1.551 s across 26 calls;
predicted masks 1.343 s; calibrated close rendering 1.553 s across 144 calls;
surface-depth processing 0.832 s. Nested times overlap and must not be added.
This motivates reducing residual native polygon work before temporal logic.

After collapsed pixel runs are handled, deduplicate only remaining byte-identical
int32 polygons via packed records. Unlike the rejected full-array structured
unique experiment, this operates on a smaller residual set. Painting identical
polygons once preserves their union exactly; no hull/bank approximation.

Sampled warm profiled native fallback: 58.10-58.36 ms, down from 69.17-71.14 ms.
Exact RGB/alpha equality in three far views; 72-pose frozen mask/box regression
recorded in `artifacts/native_residual_dedup_validation_v1`. All 29 tests pass.
Timing evidence: `artifacts/native_residual_dedup_v1`; source backup:
`frozen/pre_residual_dedup_20260912.zip`. Full combined FPS has not been rerun
after this edit. Other assets and full recorded trajectories remain pending.
Production and banks are unchanged.

## Combined Lookup and Cross-Camera Reuse

User provisionally accepted the reviewed synthetic road comparison's minor
pre-close jitter on 2026-09-12. This does not establish general camera/asset
acceptance. Close geometry remains unchanged. The experimental NativeLookup
now retains at most 16 selection results keyed by exact actor/camera matrices
and intrinsics, returning independent copies. No pose quantization or temporal
smoothing. Clear/rebuild the lookup if teacher data changes. Tests cover key
changes, copy isolation and bounded eviction; all 29 tests pass.

Repeated renderer-core mixed-distance load test (`benchmark_render_fps --lookup`)
clears caches at every camera set, so selection reuse is across cameras, not
static playback. Baseline uses fused traversal with the exact selector; the
candidate combines guarded lookup, selection reuse and fused traversal.

| Cameras | Bus instances | Baseline sets/s | Combined sets/s |
| --- | ---: | ---: | ---: |
| 1 | 1 | 21.42 | 21.47 |
| 1 | 2 | 12.11 | 13.21 |
| 1 | 3 | 6.09 | 6.74 |
| 3 | 1 | 6.92 | 11.22 |
| 3 | 2 | 3.32 | 4.98 |
| 3 | 3 | 2.22 | 3.57 |

Preliminary six-frame warmed sweep at 400x300, with synthetic scene depth;
same limitations as RENDER_FPS_RESULTS.md. Neither an independent cache/lookup
ablation nor reliable tail-latency evaluation. Shared source caches and fixed
backend ordering can bias results. Native work still dominates some poses.
No production defaults changed. Evidence: `artifacts/render_fps_lookup_v1`.

## Lookup Continuous-Pass Review

Rendered 101 poses from 40 to -10 m at 0.5 m spacing, camera z=2.3 m and
yaws 0/-60/+60, bus lateral offset -4.2 m. Synthetic gray background, no
scene depth; this isolates native-selection differences, not CARLA fidelity.
Both paths use the same fused close traversal and current calibrated bank.
Native lookup is injected only into this test and never enabled by default.

Of 303 views, 16 differ in RGB. Minimum alpha-threshold silhouette IoU versus
the exact selector is 0.9790. Maximum channel difference is 180/255; high
silhouette agreement does not imply appearance equivalence. Native metadata
reports distilled selection in 39 views; boundary blends can also contain
lookup contributions without exposing the nested native selection label.

Differences occur at frames 7-12, 55 and 59 in front/left views. Frame 55 has
7,984/11,146 changed pixels respectively and visible appearance differences.
Frame 59 carries differences into the boundary blend. Fully calibrated close
frames remain unchanged. Front selection modes switch at frames 5,16,54,56
(last enters close/blend). This is not accepted as temporally equivalent.

Video: `artifacts/native_lookup_video_v1/comparison.mp4`, verified 101 frames
and decoded last frame. Frame 55 inspected in `review_055.png`; per-view
metrics and keys in `rows.json`. All 28 unit tests pass. No default change.

Next: current-teacher refinement and stronger rejection around uncertain
native/close transitions, then repeat the continuous image gate. Do not
promote the existing lookup based solely on its selection microbenchmark.

## Existing Guarded Native Lookup Trial

Isolated `native_lookup_candidate.py` reuses the existing production selector
table loader and seven-neighbor silhouette agreement guard (threshold .935).
Bank fingerprint validation is retained. Queries outside the table's distance
and elevation envelope use the current exact selector. Projected placement
bounds come from all unique vertices of the current hull, not a learned box.
The runtime renderer, close path and asset banks are unchanged.

Tested 72 bus poses: bearings 7.5,22.5,...352.5 degrees at 12.5/25/40 m,
camera z=2.3 m, current virtual-camera aiming. Training-set overlap has NOT
been audited; these are diagnostic poses, not claimed held-out validation.
The table was reused, not regenerated from the current teacher.

- Lookup used: 47/72; exact fallback: 25/72.
- Same selected sprite as current teacher: 52/72.
- Maximum silhouette IoU loss against current teacher mask: 0.0133624.
- Exact equality of projected boxes across all 72 poses.
- Median selection-only time: 54.82 ms exact vs 10.79 ms guarded lookup.

One timing sample per pose, reference first; not a controlled FPS benchmark.
This measures selection including bounds, NOT total rendering, and neighbor
agreement is a heuristic rather than a guarantee of teacher equivalence.
Twenty differing sprite choices need appearance/temporal review. No default
backend change or promotion. Next: render complete native intervals, inspect
transitions at guard boundaries and calibrate on current-teacher data if needed.

Evidence: `artifacts/native_lookup_v1/{summary,rows}.json`.
Run: `python -m astra.benchmark_native_lookup` (new output required on rerun).

## Exact Mask-Area Score Cache

Cached immutable bank-mask areas and compute union as source area plus target
area minus intersection. This removes the repeated union popcount without
pruning candidates or approximating scores. Cache rebuilds on packed-mask
array replacement; masks remain immutable in the current runtime.

Sampled warm profiled fallback: 69.17-71.14 ms versus 72.80-73.53 ms, a modest
improvement requiring repeated unprofiled measurement for a firm speed claim.
All three far-view RGB/alpha arrays match exactly. Unit coverage compares exact
old/new IoU scores on empty, full, random and non-byte-aligned masks. All 28
ASTRA tests pass. No full renderer FPS rerun was performed for this change.

Evidence: `artifacts/native_area_cache_v1`; baseline source:
`frozen/pre_iou_area_cache_20260912.zip`. All banks remain unchanged.
Publication claims must use representative measured throughput with correctness
gates, not a target FPS or reciprocals of selected low-latency views.

## Native Integer Histogram Accumulation

Replaced four-corner axis reductions with pairwise minimum/maximum operations,
and scattered `np.add.at` pixel-run coverage updates with integer `bincount`
histograms. Polygon geometry, remaining OpenCV rasterization and ranking are
unchanged. Source backup: `frozen/pre_native_histogram_20260912.zip`.

Warm profiled native fallback now takes 72.80-73.53 ms in the three sampled
far-camera views, versus 107.48-107.74 ms before this change. Approximately
32% lower latency for this step, and about 2.3x faster than the initial 170 ms
path. This is single-image native rendering, not depth-enabled multi-camera
or multi-actor throughput. No bank reduction or production edits.

Exact RGB/alpha equality in all three far views. Evidence:
`artifacts/native_histogram_v1`; 72-pose mask/box regression:
`artifacts/native_histogram_validation_v1`. Broader multi-asset/full-sequence
validation remains a gate before promotion.

## Shared Vertex Projection

Bank reduction is deferred. Added one-time unique vertex indexing to the
static native hull. Runtime projection/normalization operates on unique
coordinates; integer polygon corners are reconstructed with the inverse
index before the existing rasterizer. Hull geometry and sprite bank are
unchanged. Cached indexing rebuilds if the face array is replaced; faces
must remain immutable in place, as in the existing runtime.

Sampled warm profiled native fallback: 107.48-107.74 ms, versus 140.60-143.03
ms after the pixel-run optimization (about 24% lower). The original sampled
path was about 170 ms. First-call indexing cost is outside these warm numbers.
This is not a depth-enabled camera-set or multi-actor throughput result.

All 72 bus mask/box comparisons against the original frozen selector are
exact. RGB and alpha match exactly for the three far-camera benchmark views.
All 27 unit tests pass. Full trajectory and other asset validation remain
pending. Banks and production files were not changed.

Evidence: `artifacts/native_vertices_v1`,
`artifacts/native_vertices_validation_v1`; source backup:
`frozen/pre_shared_vertices_20260912.zip`.

## Native Rasterization: Collapsed Pixel Runs

Retained an exact shortcut in ASTRA `Selector.predicted_mask`: projected
polygons whose integer width or height is zero are inclusive axis-aligned
pixel runs. Accumulate their union with a difference image and prefix sums;
use the original OpenCV polygon fill for all other faces. Bank coverage,
projection, scoring and sprite ranking are unchanged. Single-pixel-only
batching did not improve timing and was superseded by the run implementation.

Native fallback at the three sampled yaw views fell from approximately
170 ms to 141-143 ms (roughly 16-18%). These are profiled warm diagnostics,
not production latency guarantees. RGB/alpha match in the three far views.
An additional 72 bus poses (24 bearings at 10/20/40 m) have exactly equal
normalized predicted masks and boxes versus the frozen original implementation.
All 27 unit tests pass. Other assets/full trajectories remain unvalidated.

Evidence: `artifacts/native_singlepixel_v1`, `artifacts/native_runs_v1`,
`artifacts/native_runs_validation_v1`. Backup:
`frozen/pre_native_points_20260912.zip`. Ignore close-view timing in these
repeat-identical-pose benchmarks: those benefit from the intersection cache.

## Bank Reduction Decision

No banks were changed/deleted. A smaller deployment bank is plausible, but
native projection/rasterization cost is not proportional to sprite count.
Reducing sprites alone will not remove this bottleneck. Current initialization
also uses source masks to construct/refine hulls, so packaging a validated
precomputed hull and a selected runtime appearance subset requires an explicit
new loading/manifest contract. Keep original captures as offline evidence.

The active close banks are already compact additions, not the full dense close
capture archives. Validate pruning over held-out distances, sides, elevations,
camera intrinsics and boundary transitions, not just sprites selected on one
accepted video. A close-only deployment cannot yet replace native fallback
outside its calibrated domain. Optimize the current path before that experiment.

## Fused Complete-Render Sweep

`validate_fused_render.py` compared the reference and fused traversal inside
the same calibrated bus renderer at 31 longitudinal positions (18 through
-12 m), each with NEAT yaw 0/-60/+60, 400x300 FOV100. Both paths use exact
intersection reuse between rendering and depth. A synthetic foreground strip
tests scene occlusion. The active runtime backend was not changed.

Across all 93 views: zero alpha-threshold mask changes before or after
occlusion; maximum RGB channel difference 1/255; maximum alpha difference
6.0022e-05. A directly alongside comparison was visually inspected.

Median complete rendering plus surface-depth/occlusion time in this sweep:

| Mode | Reference ms/image | Fused ms/image |
| --- | ---: | ---: |
| Calibrated hull | 55.65 | 14.99 |
| Boundary blend | 198.37 | 191.16 |
| Native fallback | 197.49 | 171.85 |

These medians include empty camera views and are not camera-set FPS. One
sample per method/pose, fixed reference-first ordering, no model/simulation
or encoding in timing. Initial distance 18 is excluded from timing summaries;
later source-cache misses can still affect results. This is regression
evidence with diagnostic timing, not a controlled throughput benchmark.

Evidence: `artifacts/fused_render_sweep_v1/{summary,rows}.json` and sampled
comparison JPEGs. This is a synthetic pose sweep, not the full recorded
201-frame pass or 40-second NEAT replay. Other assets, depth boundaries
near the 5 cm tolerance, and full-sequence acceptance remain pending.
Native selector/rasterizer work is now the evident next speed bottleneck.

## Fused CUDA Prototype

Isolated `fused_traversal.py` / `fused_traversal.cu` assigns one CUDA thread
per ray, preserves the fixed sampling lattice, and stops at the first hit.
It is NOT connected to the runtime renderer. The trilinear interpolation is
implemented separately and is not bit-identical to PyTorch.

Nine uncached 400x300 bus views were compared to the frozen traversal, with
warmup and three timed repeats per implementation/view. Timing includes
transfers and synchronization, but excludes initial kernel compilation.

| View | Reference ms | Fused ms |
| --- | ---: | ---: |
| Approach front | 125.24 | 2.07 |
| Approach left | 195.72 | 2.38 |
| Alongside left | 296.77 | 2.02 |
| Departure left | 59.74 | 2.30 |

All nine hit masks agree exactly. Maximum hit-position coordinate difference
is 1.4782e-05 m (about 0.015 mm). Fused traversal ranges 1.73-2.39 ms across
the nine views. These are traversal timings, NOT complete renderer FPS.

Evidence: `artifacts/traversal_fused_v3/summary.json`. Earlier v1/v2 attempts
failed before executing the kernel due to missing runtime dependencies.
Reproduce with `python -m astra.benchmark_traversal --fused --output
astra/artifacts/NEW_DIRECTORY` (one command line).

Workspace-local dependencies only: CuPy CUDA12x 14.2.0, fastrlock 0.8.3 and
cuda-pathfinder 1.8.1 in `artifacts/gpu_runtime`, NVIDIA CUDA runtime/headers
12.9.79 in `artifacts/cuda_headers`. Runtime compilation uses the existing
PyTorch CUDA DLL directory. No model-environment packages were replaced.

Before enabling: validate synthetic boundary/tangent/miss rays and multiple
assets; compare rendered RGB/alpha/depth-occlusion results over complete
trajectories; measure full renderer latency. Native selection and source
sampling remain separate costs. No production promotion was performed.

## Rejected Blocked Early-Termination Experiment

`early_traversal.py` is an isolated, inactive experiment. It samples blocks
of 64 steps, retaining the previous sample at block boundaries and removing
hit rays. All nine tested bus views have exactly matching complete hit-point
and hit-mask arrays, including unused miss points.

It is substantially slower in this PyTorch implementation: approach front
140.79 -> 1872.86 ms, approach left 212.83 -> 2654.22 ms, alongside left
297.19 -> 3279.94 ms. Repeated dynamic filtering and small GPU operations
appear to outweigh reduced sampling; this causal interpretation needs a
kernel-level profile. The candidate was never enabled in `HullRays`.

Reproduce: `python -m astra.benchmark_traversal --early` with a new output
directory (the script intentionally refuses to overwrite existing evidence).
Results: `artifacts/traversal_early_v1/summary.json`. Warmup plus three timed
repeats per implementation/view, CUDA synchronization, no intersection cache.

Avoid further small-block Python/PyTorch tuning without profiling. The next
substantive path is a fused first-hit kernel or accelerated surface traversal,
with explicit checks for changes to the interpolated occupancy surface.
The accepted renderer and prior exact intersection cache remain unchanged.

## Rejected Traversal Batching Experiment

Tested reading all chunk sample counts in one GPU-to-CPU transfer rather
than one per chunk. Nine uncached bus views (forward 6/0/-6 m, yaw 0/-60/+60)
matched the frozen implementation's complete hit-point and hit-mask arrays
exactly. Three timed repeats followed warmup for each implementation/view.
However, there was no consistent speedup: approach front 122.85 -> 131.63 ms,
alongside left 287.97 -> 314.54 ms, approach left 220.77 -> 214.38 ms.
The batching change was removed; exact intersection reuse remains enabled.
All 27 tests passed during the experiment.

Evidence: `artifacts/traversal_batch_v1/summary.json`. Source baseline:
`frozen/pre_traversal_batch_20260912.zip`. The benchmark script compares the
frozen method with the current method; rerunning after rejection tests the
restored implementation, not the rejected batching code.

Next candidate should reduce actual sample work (first-hit early termination
or accelerated traversal), not merely reschedule synchronization. Any such
change needs matched depth/alpha checks at alongside and border poses.

## Depth Intersection Reuse

Implemented exact one-entry intersection reuse in `HullRays`. Source backup:
`frozen/pre_depth_cache_20260912.zip`. Cache keys compare complete origin and
direction arrays, hull identity, bounds, spacing and tensor mutation version
where available. Refinement clears the cache explicitly. Inference volumes
must not be mutated outside refinement (PyTorch supplies no version counter).
Cached outputs are read-only. Memory is bounded to one ray field and result
per hull; cameras/actors with different rays cannot share hits accidentally.

`python -m astra.benchmark_depth_cache` tests one close bus pose at native
NEAT resolution, with a controlled foreground depth strip. It alternates
cache-disabled/enabled paired rendering plus depth occlusion, discards one
warmup pair and measures five pairs per camera, synchronizing CUDA. It clears
the cache before every frame to exclude static-frame reuse. No model,
simulation, video decoding or encoding is timed.

| Camera yaw | Cache disabled median ms | Enabled median ms |
| --- | ---: | ---: |
| 0 | 268.71 | 141.56 |
| -60 | 437.98 | 232.69 |
| +60 | 25.95 | 21.14 |

RGB, alpha and depth arrays match exactly in all measured pairs. All 27
ASTRA unit tests pass, including changed rays, changed origin, changed volume
and read-only cache output checks. Raw results:
`artifacts/depth_cache_benchmark_v1/summary.json`.

This is a narrow microbenchmark, not full trajectory or multi-actor acceptance.
The visible close views remain slow: dense first-hit traversal and native
selection are still the next bottlenecks. The old repeat-identical-pose
`benchmark_candidate.py` now benefits from cross-frame cache hits; do not
compare its timings directly to earlier uncached runs.

Date: 2026-09-12. ASTRA only; not promoted to production.

## Preserved Baseline

`frozen/pre_performance_20260912.zip` preserves the working renderer source
and candidate manifest. Existing banks and scenario videos are unchanged.

## Target and Measurement

100 renderer frames/s means less than 10 ms per frame. For three cameras and
multiple actors, count a frame as the entire camera set, not one actor/view.
The recent NEAT run averaged 26.632 ms for inference alone; speeding up only
rendering cannot give that sequential full pipeline 100 FPS. Model inference,
CARLA sensor/tick time, transfers and recording require separate measurements
and optimization. No 100 FPS claim is supported yet.

`benchmark_candidate.py` measures nine warm bus views: longitudinal offsets
30, 6 and -6 m, each at camera yaw 0/-60/+60, 400x300 FOV100. It uses three
repeats per view with CUDA synchronization and cProfile enabled. This is a
small profiled diagnostic, not a production throughput benchmark or a full
trajectory correctness gate. Construction and warmup are outside timing.

Baseline outputs: `artifacts/perf_baseline_v1`.
First retained optimization: `artifacts/perf_roi_v1`.

## First Changes

Cache immutable camera rays by calibration. Do not compute the unused plane
intersection in hull mode. Restrict source projection/remapping to the
rectangle containing hull hits; skip texture sampling entirely for empty
views. Source selection and interpolation weights remain unchanged.

| Case | Baseline ms/view | Optimized ms/view |
| --- | ---: | ---: |
| Far front | 168.77 | 169.88 |
| Close front | 159.37 | 137.25 |
| Close left | 253.91 | 235.59 |
| Close right, empty | 50.74 | 13.16 |
| Departure left | 102.88 | 75.55 |

All nine RGB arrays match the baseline exactly. Eight pixels in departure-left
alpha differ by at most 5.9604645e-08; all thresholded masks match exactly.
These sampled checks do not replace the complete overtake and scenario gates.

Duplicate integer polygon removal was also measured (`perf_unique_v1`), but
increased far-view latency to about 180 ms. That change was removed.

## Remaining Work Before Benchmarking

1. Replace the native fallback's Python polygon loop with a validated faster
   projection/selection path. Profiling counted 1.42 million fillConvexPoly
   calls across nine native renders. Preserve the current selector as teacher.
2. Prototype accelerated first-hit hull traversal. Dense sampling currently
   evaluates interior samples after the first visible surface and incurs
   repeated CPU/GPU synchronization. Preserve the existing interpolated
   occupancy surface or explicitly quantify any approximation.
3. Keep ray generation, hit points and source sampling on GPU where beneficial;
   share actor-local selection across co-located cameras without sharing their
   distinct projections. Batch actors/views and bound caches.
4. Validate RGB, alpha, selection, clipping and temporal transitions across
   full bus/Tesla/Patrol passes and the three-camera scenario before promotion.
5. Measure unprofiled cold/warm p50/p95/p99 latency and total camera-set FPS
   for 1/2/3 actors, then the full pipeline separately.

Do not trade the accepted close-pass geometry for speed without a separate
candidate and explicit visual acceptance. Pedestrian retains its prior path.
