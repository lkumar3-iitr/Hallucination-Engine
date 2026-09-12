# Bus Pass-By: Calibrated Hull Candidate

Date: 2026-09-11

Test: `static_left_bus_ego_overtakes_200`, the original static-bus overtake.
All new work and outputs remain in ASTRA; production was not promoted.

## Protocol

Re-rendered the complete frozen CARLA sequence against its existing clean HE
background, without starting a new CARLA simulation or using inpainting.
Camera calibration, FPS, frame identities, actor poses, and camera poses were
checked before rendering (pose tolerance 1e-4). The reference uses 201 frames
at 20 FPS, 1280x720, FOV90, camera mount (1.5, 0, 1.6), and 3.5 m lateral
actor offset. Runtime GT masks are used only after rendering for evaluation.

Candidate: calibrated capture sampling, refined bank-only hull, and continuous
source interpolation from `artifacts/bus_aimed_close_v3`. All finite bank
boundaries use a smooth transition to the native baseline.

## Results

| Measurement | Earlier accepted hybrid | New candidate |
| --- | ---: | ---: |
| Frames | 201 | 201 |
| CARLA-visible frames | 72 | 72 |
| Mean visible-frame mask IoU | 0.9099 | 0.9573 |
| Visibility agreement | 100% | 100% |
| Dropouts | 0 | 0 |
| Empty-GT frames with HE pixels | 0 | 0 |

Candidate frame 50 has mask IoU 0.9858. The close-pass body looks rigid in
the inspected frames. The final mirror fragment is still imperfect: frame
70 has IoU approximately 0.647, versus approximately 0.116 in the earlier
accepted result. This is not a claim of pixel-perfect rendering. Source
illumination, reflections, and shadow differences remain.

All 19 ASTRA unit tests pass; the modified evaluation module compiles.

## Evidence

[Comparison video: CARLA, candidate, masks](artifacts/static_left_bus_overtake_calibrated_hull_v1/comparison.mp4)

- Candidate output: `artifacts/static_left_bus_overtake_calibrated_hull_v1/`
- CARLA reference: `artifacts/static_left_bus_overtake_carla_v2/`
- Clean background: `artifacts/static_left_bus_overtake_he_v8_horizontal_fullrange/`
- Earlier metrics: `artifacts/static_left_bus_overtake_comparison_v8_horizontal_fullrange/summary.json`

The candidate output includes every frame, source-selection metadata, true-alpha
mask metrics in `rows.csv`, and `summary.json`.

## Reproduce

From the repository root in the `he_neat` environment, use a new output path:

```powershell
python -m astra.evaluate_calibrated_close --reference astra/artifacts/static_left_bus_overtake_carla_v2 --background astra/artifacts/static_left_bus_overtake_he_v8_horizontal_fullrange --close-bank astra/artifacts/bus_aimed_close_v3 --native-bank D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/fuso_rosa_bus_native_full_v3 --output astra/artifacts/my_bus_passby_test --hull --refine-hull
```
