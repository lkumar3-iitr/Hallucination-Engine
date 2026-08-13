# Frozen HE Placement / Rendering v1

**Status:** FROZEN / VALIDATED

This package freezes the HE v1 placement and rendering configuration after the standardized CARLA–HE geometric validation.

No placement lookup tuning is performed by this finalization step.

## Frozen camera

- Resolution: 1280 x 720
- Horizontal FOV: 90.0 deg
- Mount: x=1.5 m, y=0.0 m, z=1.6 m
- Orientation: pitch=0 deg, yaw=0 deg, roll=0 deg
- Intrinsics: fx=640.0, fy=640.0, cx=640.0, cy=360.0

## Canonical controlled benchmark

- Controlled cut-in cases: 8
- Factors: start distance, actor speed, cut-in duration
- HE placement OOD checking enabled
- Exact CARLA–HE backend state invariant validated before execution

## Depth-stratified geometric result

| Minimum depth | Cases | BBox frames | cx MAE | bottom-y MAE | width MAE | height MAE |
|---:|---:|---:|---:|---:|---:|---:|
| >= 15 m | 8/8 | 1888 | 2.063 | 3.157 | 4.261 | 2.881 |
| >= 20 m | 8/8 | 1609 | 3.052 | 2.644 | 3.694 | 2.397 |
| >= 25 m | 6/8 | 1314 | 1.496 | 2.053 | 2.831 | 1.849 |
| >= 30 m | 6/8 | 694 | 2.286 | 1.721 | 2.259 | 1.542 |

The scale and vertical-placement errors decrease as near-field frames are excluded. Horizontal-center error is not expected to be strictly monotonic because it also depends on lateral position and viewpoint.

## Frame-weighted MAE

| Minimum depth | cx | bottom-y | width | height |
|---:|---:|---:|---:|---:|
| >= 15 m | 2.053 | 3.091 | 4.169 | 2.819 |
| >= 20 m | 2.351 | 2.339 | 3.180 | 2.114 |
| >= 25 m | 1.530 | 2.012 | 2.764 | 1.812 |
| >= 30 m | 2.382 | 1.496 | 1.968 | 1.336 |

## Validation coverage

- Controlled cut-in benchmark: standardized quantitative validation.
- Oncoming scenario: standardized V2/M4 execution validation.
- Multi-actor scenario: V2 multi-actor execution validation.
- Static, following/receding, crossing, ego-turning and earlier cut-in/oncoming cases remain engineering validation evidence from earlier pipeline stages.

## Frozen placement operating domain

- Camera-relative lateral x: [-4.0, +4.0] m
- Camera-relative forward z: [0.0, 100.0] m
- Actor relative yaw: [0, 359] deg
- z <= 0: actor is behind camera and culled.
- Placement outside the lookup domain must be reported as PLACEMENT_OOD rather than silently treated as valid.

## Known limitations

- Near-field scale and vertical-placement error is larger than far-field error.
- Partial near-camera clipping is not modeled as a separate visibility state.
- General scene occlusion is not fully modeled.
- Frozen lookup validity is limited to its defined camera-relative placement domain.
- These limitations are documented rather than corrected by fitting the lookup to validation cases.

## Freeze decision

The placement lookup, camera convention, viewpoint calculation, coordinate transforms and V2 rendering pipeline are frozen after this validation.
