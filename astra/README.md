# ASTRA: isolated sprite selection

## Scenario 8 Candidate (2026-09-11)

The isolated bus repair, three-camera evidence, and full 40-second CARLA/HE
comparison are documented in [SCENARIO_08_CALIBRATED_HULL.md](SCENARIO_08_CALIBRATED_HULL.md).
This is an experimental candidate, not a replacement for the accepted checkpoint
below. Production files have not been changed by this repair.

The original static-bus overtake was also replayed with the candidate; see
[BUS_PASSBY_CALIBRATED_RESULTS.md](BUS_PASSBY_CALIBRATED_RESULTS.md).

Compact Tesla, Patrol, and pedestrian tests and their storage inventory are in
[COMPACT_MULTI_ASSET_RESULTS.md](COMPACT_MULTI_ASSET_RESULTS.md). The pedestrian
candidate regressed and remains disabled; the previous backend is retained.

Accepted renderer configuration: `astra_passby_v1_close_hybrid`. See
[CURRENT_HE_STATE.md](CURRENT_HE_STATE.md) for the concise project status and
accepted scenario metrics.

The production 4320-view banks remain the general-view source. Opt-in,
asset-specific Cartesian-close banks cover the difficult side-pass interval;
see `tesla_sidepass_asset_manifest_v1.json`,
`patrol_sidepass_asset_manifest_v1.json`, and
`pedestrian_sidepass_asset_manifest_v1.json`. The immutable pre-optimization
checkpoint is under `frozen/`, and performance evidence is in
[OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md).

For the complete side-pass renderer and no-GT overtaking videos, see
[SIDE_PASS_RESULTS.md](SIDE_PASS_RESULTS.md). The selector-only protocol and
its original limitations are documented below; the new wrapper handles the
native camera-plane crossing with an object-centered virtual camera.

ASTRA chooses an existing bank sprite from actor/camera poses and intrinsics.
It implements the selector part of the user's decomposition: an external
boxFinder supplies the full target box and a compositor clips the sprite.
Everything created for this experiment lives in `astra/`.

## Result

The bank-only selector nearly reproduces an exhaustive silhouette oracle on
the tested poses. It does not need runtime CARLA masks, learned trajectory
labels, a CARLA mesh, a new sprite bank, or the production HE renderer.

| Evaluation, fully visible frames | Count | ASTRA IoU | Exhaustive oracle IoU |
| --- | ---: | ---: | ---: |
| Three existing recordings | 64 | 0.95028 | 0.95152 |
| Existing recordings, distance below 10 m | 33 | 0.94374 | 0.94516 |
| Independent CARLA poses | 74 | 0.94350 | 0.94526 |
| Independent poses, distance below 10 m | 42 | 0.93914 | 0.94152 |

These are mean IoUs after independent width/height normalization to 128x128,
not full-image placement scores or RGB similarity scores. Every one of the
138 evaluated cases is within 0.02 IoU of its exhaustive 4320-candidate oracle.
The maximum gaps are 0.01000 on existing recordings and 0.01093 on new poses.
Native-box selected-mask mean IoUs are 0.94989 and 0.94300 respectively.
The simple nearest angle/distance/elevation baseline scores 0.91813 and 0.88752;
it is an ASTRA baseline, not a rerun of the current production renderer.

## How It Works

1. Read the exact source actor/camera transforms, intrinsics, crop offsets,
   and alpha masks from the existing bank.
2. Carve a 3.5 cm voxel volume using 144 existing silhouettes: 36 bearings,
   four elevations, and the 5 m source slice. Only bank evidence enters here.
3. Project the reconstructed surface into the runtime camera. This accounts
   for lateral position, perspective, elevation, yaw, pitch, and roll through
   the full transform, instead of treating a vehicle as a flat box.
4. Normalize the predicted full outline and rank the original bank alpha
   masks by IoU. A 35-degree bearing neighborhood excludes opposite-facing
   sprites that can have misleadingly similar silhouettes; a small angular
   tie-break favors the physically nearer view.
5. Return a raw bank sprite index/path. BoxFinder and the compositor remain
   separate from selection.

This is a [visual-hull reconstruction](https://doi.org/10.1109/34.273735)
used as a selection proxy. The proxy is never used as the vehicle's RGB asset.
No neural model or runtime simulator renderer is required.

## Files and Commands

- `selector.py`: reconstruction, mask-free `Selector.select`, and independent
  `composite_to_box` with returned alpha.
- `evaluate.py`: exhaustive canonical-IoU oracle and frame-by-frame reports.
- `capture_validation.py`: isolated CARLA capture with frame matching, instance
  calibration, owned-actor cleanup, and restoration of world settings.
- `preview.py`: RGB comparisons and optional verified clean-background preview.
- `test_selector.py`: projection, transform invariance, normalization, alpha,
  and clipping-contract tests.
- `RESULTS.md`: reproducibility, evidence paths, and limitations.

From the repository root, using the existing Python environment with NumPy,
OpenCV, and CARLA installed:

```powershell
python -m unittest discover -s astra -p test_selector.py
python astra/evaluate.py --output astra/artifacts/reproduced_existing
python astra/evaluate.py --references astra/artifacts/carla_independent_v4 --output astra/artifacts/reproduced_independent
```

To capture another independent set on an idle CARLA server, use a new output
directory. The recorder refuses to overwrite existing captures:

```powershell
python astra/capture_validation.py --output astra/artifacts/carla_new_validation
```

API, using CARLA transforms as pose containers (no live server needed):

```python
from astra.selector import Selector, composite_to_box
import cv2

selector = Selector(bank_path, "astra/artifacts/cache")
selector.build_hull()
choice = selector.select(actor_tf, camera_tf, 1280, 720, 90.0)
sprite = cv2.imread(choice["sprite_path"], cv2.IMREAD_UNCHANGED)
image, alpha = composite_to_box(background_bgr, sprite, full_target_box)
```

`full_target_box` is `(x1, y1, x2, y2)` with exclusive end coordinates. It must
describe the entire sprite before viewport clipping. Fitting the whole sprite
to a clipped visible box is a different operation and is incorrect here.

## Scope

The reported selector result holds scale and placement fixed using GT boxes,
and evaluates fully visible, unoccluded silhouettes. The API does not accept
GT masks or GT boxes for selection. The clean-background preview uses GT boxes
solely to stand in for boxFinder; it is not an end-to-end no-GT placement claim.

Lighting, reflections, RGB fidelity, external occlusion, and multi-actor
overlap remain separate issues. The centered-camera renderer and targeted
close banks handle the validated near-plane and viewport-clipped pass-by
cases. Existing-bank source capture calibration is required. Matched authored
tests validate Tesla and Patrol rigid-vehicle rendering; pedestrian rendering
is continuous but remains experimental at 0.8060 mean mask IoU. Exact angular
prefiltering and packed-mask IoU reduce Tesla native selection from about 125
ms to 35-38 ms per tested pose on this machine. Complete rendering is not yet
real-time.

The lowest independent selected IoU is 0.891; its dataset oracle minimum is
0.892. Near-oracle selection does not mean a perfect raw sprite exists for
every possible camera configuration.
