# Renderer Logic and Optimization Map

Status: 2026-09-12. A reasoning guide, not a claim of universal validation.
Scope: the current calibrated ASTRA renderer, its exact reference selector,
and isolated speed candidates. Production `he_renderer` has not been replaced.

## 1. Are These General Optimizations?

Yes in mechanism: runtime decisions use actor/camera geometry, calibration,
bank data, mask coverage and cache keys. They do not use scenario IDs, frame
numbers, a known overtake time, or CARLA target masks to choose a shortcut.
Benchmark scripts intentionally contain specific poses; those are test inputs,
not runtime rules.

However, distinguish these three statements:

- General algorithm: applicable to compatible geometry/calibration inputs.
- Asset-specific data: each asset still needs its own calibrated bank/hull;
  a bus lookup cannot be used for a Tesla.
- Validated domain: recent speed/quality tests concentrate on the bus and
  400x300 FOV100 cameras. General correctness/performance across all assets,
  poses, intrinsics, occlusions and interactions has not been demonstrated.

Exact raster shortcuts have algebraic reasons for preserving output under
their stated assumptions. The fused CUDA sampler and guarded lookup require
empirical error checks; neither is an exact-equivalence theorem.

## 2. Inputs and Outputs

Per camera/frame:

- Background RGB without physical HE adversaries.
- Camera world pose, width/height and horizontal FOV.
- Virtual actor identity and world pose, plus its calibrated asset metadata.
- Optional aligned scene depth in metres, same camera pose/time/intrinsics.

Persistent asset data:

- Native RGBA sprites, capture transforms/intrinsics and physical bounds.
- Compact close RGBA captures with full calibration and crop offsets.
- A bank-derived visual hull; optionally a native selector lookup table.

Outputs: composited RGB, actor visibility/placement metadata, and internally
alpha and optional surface depth. Runtime CARLA target RGB, target instance
masks and target boxes are not rendering inputs. Masks used offline to build
the bank/hull are different from runtime target ground truth.

The renderer does not run the driving model or simulate physical collisions.
The current implementation uses CARLA transform conventions/APIs even when
rendering offline; it is not a fully simulator-independent implementation.

## 3. Unoptimized Reference Logic

Here "unoptimized" means a straightforward implementation of the current
calibrated geometry, before the recent speed shortcuts. It does NOT mean the
older planar renderer with the previously observed close-pass bending.

### 3.1 Offline preparation

1. Normalize each native alpha silhouette to a common mask resolution and
   pack its bits for comparison.
2. Build a coarse visual hull by carving a physical-bounds voxel volume with
   calibrated bank silhouettes, then extract exposed voxel faces.
3. For the calibrated close path, reconstruct occupancy from these faces and
   refine it with the complete calibrated close-bank masks.
4. Retain capture matrices and intrinsics so runtime 3-D points can be mapped
   back into each source image, accounting for its crop origin.

The visual hull is an approximate shape inferred from silhouettes, not the
original CARLA triangle mesh. It does not recover unseen concavities or
view-dependent appearance. Existing initialization already caches expensive
offline products on disk and decoded close images in a small source cache.

### 3.2 Coordinate query and branch choice

Let A map actor-local coordinates to world coordinates, and C map native
camera coordinates to world coordinates. Then:

```text
T = inverse(A) * C
o = translation(T)             # camera origin in actor coordinates
q = hull_center - o            # close-bank query
```

The close bank is parameterized by signed longitudinal/lateral offsets and
vertical offset. Bracket q on these axes. Never interpolate between captures
on opposite sides through the actor. If required support is absent, use native
fallback. This is a coverage decision, not simply "distance below N metres."

### 3.3 Native fallback: predict outline, select, place

1. Construct a virtual camera at the SAME origin as the native camera.
   In the current `PassbyRenderer`, aim toward the closest point on the actor's
   physical box (not always its center). A camera inside that box is unsupported.
2. Project all visual-hull face corners into this virtual camera. Require
   positive depth beyond the selector's near-plane threshold.
3. Compute the full projected bounds B from coordinate minima/maxima.
4. Normalize projected coordinates inside B to a fixed mask resolution;
   round coordinates and rasterize every polygon to obtain predicted mask M.
5. Compute query bearing, center distance and elevation. Restrict candidate
   sprite bearings to within 35 degrees, with circular angle differences.
6. Compare eligible native alpha masks with M using silhouette IoU and a small
   bearing penalty. Choose the maximum-ranked sprite. The reference already
   searches an eligible subset, not all bank sprites every frame.
7. Fit the sprite's alpha crop into full, unclipped bounds B.
8. Inverse-map native pixels into the virtual image by pure camera rotation.
   Sample premultiplied RGBA; discard invalid rays/out-of-image samples.

For identical intrinsics K, the native-to-virtual image mapping is a rotation
homography, with coordinate-axis conversion included by the implementation:

```text
pixel_virtual ~ K * R_virtual^-1 * R_native * K^-1 * pixel_native
```

This preserves viewport clipping without making whole-object visibility depend
only on actor-center depth. It is still single-sprite appearance reprojection;
the calibrated close path is what improves strong near-field parallax.

### 3.4 Calibrated close path: ray hit, source sampling, blending

For each native pixel (u,v), with focal length f and principal point (cx,cy):

```text
d_camera = [1, (u-cx)/f, -(v-cy)/f]
d_actor  = rotation(T) * d_camera
ray(t)   = o + t * d_actor
```

These directions are not normalized. The reference traversal:

1. Intersect the ray with the hull volume's axis-aligned bounds.
2. Sample between entry and exit with world-space spacing half a voxel:
   `delta_t = voxel_spacing / (2 * norm(d_actor))`.
3. Evaluate trilinearly interpolated occupancy at all sample points.
4. Find the first sample with occupancy >= 0.5 and valid ray depth.
5. Interpolate between that sample and its predecessor to estimate the hit.

For each hit point X, select up to eight neighboring capture nodes by the
three query-axis brackets. For each source j:

```text
X_source = inverse(C_capture_j) * A_capture_j * X
u_source = cx_j + fx_j * X_source.y / X_source.x - crop_x_j
v_source = cy_j - fy_j * X_source.z / X_source.x - crop_y_j
layer    = sum_j(weight_j * sampled_premultiplied_RGBA_j)
```

Invalid source depth or pixels outside the source return transparent samples.
Interpolation weights come from geometry, not previous-frame selection.
Full-frame inverse sampling naturally clips to the native viewport.

### 3.5 Native/close boundary blend

The calibrated bank has finite support. Near a coverage boundary, compute a
smoothstep weight from distance to its longitudinal, lateral and vertical
limits, and multiply the axis weights. Blend premultiplied layers:

```text
layer = w * close_layer + (1-w) * native_layer
```

At full close support, native rendering is unnecessary. Near boundaries both
paths may run, so these frames can cost more. Exact versus lookup selection
differences can also enter the boundary blend.

### 3.6 Compositing and optional scene occlusion

For a premultiplied layer L with alpha a:

```text
RGB_out = L.rgb + (1-a) * RGB_background
```

Scene depth comes from the existing depth camera or an equivalent aligned
source. HE actor depth is needed separately because virtual actors are absent
from that depth image. Transform hull hit points into the native camera and
take the camera-forward coordinate, NOT Euclidean ray length.

```text
hidden = valid_scene_depth AND valid_actor_depth
         AND (scene_depth + 0.05 m < actor_surface_depth)
```

The current ASTRA implementation restores the pre-actor background and zeros
alpha at hidden pixels. Unknown actor depth and invalid scene depth leave
pixels unchanged and are counted. The 5 cm margin is not an accuracy guarantee.
No supplied depth means no scene-depth occlusion.

Actors are composited in distance order. A shared per-pixel HE-to-HE z-buffer
is NOT implemented; physical scene depth cannot resolve virtual-bus versus
virtual-pedestrian ordering. The retained legacy pedestrian close path uses
its older nearest-depth approximation rather than calibrated surface depth.

## 4. Where the Work Goes

Notation: N = eligible native sprites, V = hull face corners, U = unique hull
vertices, P = pixels per camera, R = rays intersecting hull bounds, S = samples
per active ray, K = contributing close captures (at most 8), M = mask pixels.

| Work | Straightforward scaling | Important qualification |
| --- | --- | --- |
| Native projection | O(V) | Does not shrink just by reducing sprite count |
| Native rasterization | Face count plus painted pixels | Many tiny/duplicate polygons |
| Native scoring | O(N*M) in packed units | Already bearing-restricted |
| Native warp | O(P) | Plus sprite decoding if uncached |
| Dense close traversal | O(R*S) | Samples behind first surface are wasted |
| Close appearance | O(K*P) | Source decoding/cache misses also matter |
| Scene occlusion | O(P), plus surface computation | Naive implementation traverses twice |
| Multiple actors/cameras | Approximately additive | Some exact work is shareable |

Lookup avoids outline rasterization only on accepted queries; exact projected
bounds still cost work. A 2 ms traversal is NOT a 2 ms renderer.

## 5. Retained Exact or Near-Exact Speed Changes

These are present in ASTRA source, not promoted repository-wide.

| Change | What it avoids | Correctness condition / caveat |
| --- | --- | --- |
| Cached camera rays | Rebuilding pixel rays | Same resolution/FOV; read-only arrays |
| Remove unused plane calculation | Computing a plane hit in hull mode | Hull branch supplies the actual hits |
| Hit-region source sampling | Project/remap transparent regions | Only skip outside all valid hull hits; tiny floating alpha differences observed |
| Exact intersection reuse | Second traversal for scene depth | Exact origin/rays and unchanged hull; bounded one-entry cache |
| Collapsed pixel runs | Polygon calls for horizontal/vertical lines and points | Same integer raster union |
| Unique hull vertices | Repeated projection of shared corners | Reconstruct original polygon indices afterward |
| Pairwise bounds and integer histograms | Reductions/scattered accumulation | Same integer corner extrema and coverage counts |
| Precomputed native mask areas | Repeated union popcount | Union = source area + target area - intersection |
| Residual polygon deduplication | Drawing identical remaining polygons | Byte-identical corners paint the same pixels |
| Covered-polygon elimination | Drawing polygons over fully painted rectangles | Entire inclusive bounding rectangle already covered |

Static face/mask caches assume arrays are not mutated in place. Replacing the
arrays rebuilds the relevant cache. Hull refinement explicitly clears its hit
cache; inference-mode tensors do not expose an ordinary mutation counter.

Most of these reduce constants rather than changing worst-case asymptotic
complexity. They are useful but do not eliminate native projection/rasterization.

## 6. Isolated Candidates, Not Default Runtime

### Fused CUDA first-hit traversal

One CUDA thread handles a ray and exits at its first surface hit. Same fixed
sampling lattice, separately implemented trilinear occupancy sampler; avoids
Python-managed ray batches and much post-hit work. Not bit-identical.

Nine bus views: identical hit masks, maximum hit-coordinate difference about
0.015 mm. A 93-view full-render sweep had no threshold-mask changes before or
after synthetic depth occlusion, maximum RGB channel difference 1/255.
These are limited tests, not universal guarantees. It remains explicitly
injected by experiment scripts.

### Guarded native lookup

Existing bank-specific teacher table indexed by bearing/log-distance/elevation.
Nearest-neighbor labels propose a sprite; seven-neighbor silhouette agreement
must exceed .935, and distance/elevation must lie within the table envelope.
Otherwise run the current exact teacher. Current hull vertices still supply
exact bounds. No runtime target ground-truth mask is used.

This is approximate selection: neighbor agreement is NOT a bound on image
error. The reused table was not rebuilt from the current teacher; training
overlap of diagnostic poses is unknown. In 72 poses, 20 choices differed.
The continuous review exposed small pre-close appearance/jitter differences;
the user provisionally accepted the shown video, not all possible use cases.

### Exact cross-camera selection cache

Up to 16 results keyed by exact actor and virtual-camera matrices plus
intrinsics. Return independent copies. Useful when co-located native cameras
produce identical virtual queries. Do not share native pixel projections.
Different origins/calibrations need their own results. Not temporal coherence:
no pose rounding, history-based candidate restriction or hysteresis.

The cache currently lives in the experimental lookup wrapper and can also
wrap the exact selector for ablation. Rebuild it when teacher/table data changes.

## 7. Rejected or Deferred Experiments

| Experiment | Outcome |
| --- | --- |
| Deduplicate all integer polygons using structured array unique | Slower; removed. Later residual-only packed deduplication helped |
| Batch GPU sample-count reads | Exact intersections but inconsistent speed; removed |
| PyTorch 64-sample blocks with early ray removal | Exact tested output but much slower; inactive experiment |
| Single-pixel-only polygon batching | Little gain; superseded by axis-aligned runs |
| Temporal coherence / previous-view neighborhood | Discussed, not implemented; deferred by user |
| Bank reduction | Deferred; no banks removed |
| Loosen lookup confidence for more hits | Not accepted as a speed strategy |

## 8. Interpreting the Numbers

The latest sampled native fallback is about 47-49 ms, versus the original
roughly 170 ms. These warm profiled single-image diagnostics do NOT include
the complete depth-enabled multi-camera adapter pipeline.

The last combined mixed-distance benchmark, BEFORE covered-polygon elimination,
reported approximately 22.0/13.7/7.0 single-camera FPS for 1/2/3 bus instances.
It used fused traversal plus guarded lookup/cache, synthetic depth, six poses
repeated three times, 400x300 FOV100. It is not the default backend and was
not rerun after the latest edit. Excludes model, CARLA, decoding/encoding and
adapter logging; fixed method order/shared source caches can bias comparisons.

Do not multiply speedups from different experiments. Use total images divided
by elapsed time for throughput. Three-camera sets/s is each camera's update
rate; aggregate images/s is three times that value. Per-component microbenchmarks
cannot establish final renderer FPS or paper-grade behavioral equivalence.

## 9. Questions for Exploring a Better Algorithm

- Can the native outline be generated without rasterizing all projected faces,
  while matching its integer contour and full placement bounds?
- Can offline data replace expensive online work with a validated fallback,
  including unseen poses rather than just a known test sequence?
- Can the same geometric result serve both appearance sampling and occlusion
  without copying/recomputing its representation?
- Which operations are truly necessary for an off-screen actor? Visibility
  tests must consider the whole shape, not just its center.
- How much cost is loading/CPU-GPU transfer versus computation? Keep cold
  initialization, warm rendering and video I/O measurements separate.
- Can a change improve exact fallback directly, avoiding lookup jitter entirely?

These are prompts for design, not already implemented solutions.

## 10. Code and Evidence Map

- `calibrated_bus_renderer.py`: branch selection and boundary compositing.
- `calibrated_asset_renderer.py`: native/close asset binding.
- `calibrated_close.py`: calibrated capture queries and appearance sampling.
- `hull_rays.py`: reference traversal, carving and hit cache.
- `passby_renderer.py`: native virtual camera, full bounds and reprojection.
- `selector.py`: native hull construction, mask rasterization and ranking.
- `scene_depth.py`, `calibrated_compositor.py`: depth and multi-actor integration.
- `fused_traversal.py`, `fused_traversal.cu`: isolated GPU candidate.
- `native_lookup_candidate.py`: guarded table and exact selection cache.
- [Performance history](PERFORMANCE_WORK.md): chronological experiments/evidence.
- [FPS methodology/results](RENDER_FPS_RESULTS.md): workload and timing limits.
- [Depth contract](DEPTH_OCCLUSION.md): occlusion behavior and limits.
- `frozen/`: preserved source checkpoints; not all are full dependency bundles.

No renderer or benchmark changes were made while creating this guide.
