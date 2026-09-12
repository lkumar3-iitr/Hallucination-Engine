# ASTRA Selection Experiment, 2026-09-07

Update 2026-09-08: [SIDE_PASS_RESULTS.md](SIDE_PASS_RESULTS.md) records the
subsequent complete overtaking experiment, including automatic placement,
clipped frames, clean backgrounds, and the recorder adapter. The original
selector-only results below remain as the historical first experiment.

## Question and Decision

Can we select the bank sprite that brute-force CARLA silhouette matching finds,
assuming target box dimensions, placement, and clipping are handled separately?

The tested answer is yes, to a small measured oracle gap. Build a bank-only
visual hull, project it from runtime poses, and rank existing sprite silhouettes.
The current result warrants integrating this selector experimentally with a
separately validated boxFinder. It does not warrant claiming full renderer
equivalence or perfect photorealism.

## Inputs and Protocol

- Bank: `D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3`.
- 4320 sprites: 360 bearings x 3 distances (5/10/20 m) x 4 elevations (0/5/10/20 deg).
- Bank identity: `96b6f90ccad7676ccf5860e1d8e840c91649ecbee5225e412d586bf9a3cdd7dc`.
  This hashes CSV/metadata contents and PNG size/mtime identity; it is not a
  content hash of every PNG.
- Reconstruction: 144 existing 5 m captures, bearing step 10 degrees,
  1 source-pixel dilation, 0.035 m voxel spacing; 203483 retained voxels,
  29232 exposed quad faces. Cache records reconstruction configuration.
- Selector configuration was fixed before independent evaluation. No runtime
  CARLA mask or oracle winner was used to reconstruct the hull or fit parameters.
- Oracle: exhaustive IoU over all 4320 alpha crops at 128x128. This differs
  from the earlier web oracle's Chamfer coarse search/native refinement and
  should not be described as an exact rerun of that protocol.
- Evaluation: selection occurs before mask loading; GT subsequently supplies
  an external box and normalized target silhouette. Each candidate gets the
  same box. Empty/small, border-touching, and near-plane-invalid cases are
  explicitly excluded and listed in each summary.

## Existing Recording Results

`artifacts/evaluation_full_v1/summary.json` and `rows.csv`:

| Subset | N | ASTRA | Oracle | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| All eligible frames | 64 | 0.950281 | 0.951521 | 0.001241 |
| Center distance below 10 m | 33 | 0.943744 | 0.945165 | 0.001420 |

References are the forward pass-by, camera yaw -60 degrees, and camera yaw
-80/pitch -10 degrees under `web/carla_reference`. All eligible frames were
evaluated at frame step 1; this is not a sample of only favorable frames.
Maximum oracle gap: 0.009999. All 64 are within 0.02 of oracle.

## Independent CARLA Validation

Accepted capture: `artifacts/carla_independent_v4`, with `COMPLETE.json`.
The three earlier folders are failed capture attempts and are not evidence.
The failures exposed a stale initial actor transform before world tick and
the need to calibrate annotation instance keys instead of assuming actor IDs.
The final recorder waits for a snapshot and tracks the calibrated target key.

Recorded 96 poses, with 74 fully visible eligible cases. No labels from this
capture train or tune the model. Grid: bearings 13/57/101/147/193/237/281/327
degrees, center distances 4/7/14 m, elevations 7/16 degrees, and optical yaw
offsets -18/+18 degrees. Camera 1280x720, horizontal FOV 90 degrees. Actor is
isolated at altitude 80 m; rendering annotations are frame-matched. Background
vehicle annotations are excluded using the stable target instance key.

`artifacts/independent_evaluation_v1/summary.json` and `rows.csv`:

| Subset | N | ASTRA | Oracle | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| All eligible poses | 74 | 0.943502 | 0.945256 | 0.001754 |
| Center distance below 10 m | 42 | 0.939139 | 0.941517 | 0.002378 |

Maximum oracle gap: 0.010930. All 74 are within 0.02 of oracle. Native-box
selected mean IoU: 0.942999. Predicted hull versus target normalized mean IoU:
0.974849. Runtime mean: 0.121 seconds per selected pose, excluding startup.

Resolution sensitivity check: `artifacts/independent_resolution_check` uses
256x256 masks and every fifth captured pose (14 eligible). ASTRA mean 0.945167,
oracle 0.946391, gap 0.001224; all within 0.02. This supports the result beyond
the original 128-pixel normalization, but is not a full native-resolution oracle.

CARLA cleanup was verified: asynchronous settings restored and zero spawned
vehicle/sensor actors remaining. The map and weather were not changed.

## Visual Evidence

- `artifacts/preview_forward/clean_background_gt_box.mp4`: CARLA versus ASTRA
  on the existing background-only CARLA recording, without inpainting.
  Intrinsics and per-frame camera poses are checked before using that video.
- `artifacts/preview_forward/rgb_selection_comparison.mp4`: enlarged RGB crops.
- `artifacts/preview_close_yaw60/rgb_selection_comparison.mp4`: close-range RGB.
- `artifacts/preview_independent/rgb_selection_comparison.mp4`: new-pose RGB.
- Each evaluation directory also has normalized silhouette comparison videos,
  diagnostic PNGs, frame scores, selected keys, oracle keys, and skip reasons.

RGB videos play eligible frames at 10 fps; manifests list exact frames and
warn when frame gaps exist. They use GT boxes, not predicted placement.
Inspected examples show strong outline agreement, but visible differences in
lighting, highlights, window appearance, and internal proportions. Those are
not measured by silhouette IoU and remain outside the solved selector metric.

## Verification and Next Boundary

Seven unit tests passed: CARLA axes, world-transform invariance, canonical
normalization, IoU identity, viewport clipping without rescaling fragments,
alpha correctness on equal-colored backgrounds, and invalid/offscreen boxes.

Before production integration: validate the external boxFinder independently,
measure partially clipped and occluded cases with a valid unoccluded-box
contract, assess RGB/temporal quality, and optimize runtime. Near-plane-crossing
poses currently return an explicit error. No files outside `astra/` were edited
for this experiment; no commit or push was performed.

Method reference: A. Laurentini, The Visual Hull Concept for Silhouette-Based
Image Understanding, IEEE TPAMI 16(2), 1994, DOI: 10.1109/34.273735.
