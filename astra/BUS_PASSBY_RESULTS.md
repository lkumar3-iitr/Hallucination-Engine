# Bus Static-Overtake Results

Status date: 2026-09-08

## Bank Facts

The Fuso Rosa native bank contains 5,760 sprites, not 4,320: 360 azimuths,
four distances (10, 15, 20, and 25 m), and four elevations (0, 5, 10, and 20
degrees). Its metadata physical box is approximately 10.273 x 3.944 x 4.253 m.

Exhaustive native-bank search was evaluated on the 27 fully visible poses in
the authored pass. Its mean normalized silhouette IoU is 0.9518, while ASTRA
selection reaches 0.9493. The mean oracle gap is 0.0025 and every pose is
within 0.02. This supports the native selector for ordinary fully visible
views.

It does not mean that brute force can always find an exact close-pass sprite.
The nearest native capture is 10 m, while the bus is 10.273 m long. During the
alongside pass, different parts of the bus therefore have strongly different
depths. Scaling and warping one >=10 m planar capture cannot reproduce that
near-field perspective exactly, regardless of which of the 5,760 sprites is
selected. This is the source of the apparent bent side.

The general geometry-filtered Cartesian-close bank contains 48,956 sprites.
Its capture target-up value (0.176 m) does not match this scenario's bus-center
offset (0.575 m), however, which makes the bus appear pitched upward and gives
poor tire contact. A targeted 252-sprite bank was therefore generated at
forward positions 0.25-10.5 m, lateral offsets +/-3.0, +/-3.5, and +/-4.0 m,
and relative yaw 0 degrees. All camera and actor transforms remain level. The
geometry-aware manifest accepts all 252 positions and the dataset has no
missing or extra files.

## Hull Correction

The original hull builder was hard-coded to carve from 5 m captures. That
worked for Tesla, Patrol, and pedestrian, but the bus CSV has no 5 m rows. The
first bus result therefore used the uncarved physical box rather than a visual
hull.

Hull cache version 2 selects the nearest distance actually present in each
bank. For the bus this is 10 m. The resulting bank-only hull uses 132 valid
captures, 2,704,719 occupied voxels, and 157,870 surface faces.

The virtual camera now aims toward the closest point on the physical box. For
an outside camera this is a separating direction, placing the containing box
in positive virtual depth. The exact same-origin rotation into the native
camera is unchanged. Physical camera/actor intersection remains unsupported.

## Authored Scenario

`static_left_bus_ego_overtakes_200` changes only the accepted Tesla scenario's
actor identity and metadata dimensions. Ego motion, static actor trajectory,
camera, timing, and 3.5 m adjacent-lane offset are unchanged. Both conditions
contain 201 frames at 20 fps, 1280x720, FOV 90, with camera `(1.5, 0, 1.6)`.

| Metric | Result |
| --- | ---: |
| Frames | 201 |
| Both visible | 72 |
| Visibility agreement | 1.0000 |
| Mean bbox IoU | 0.9643 |
| Mean mask IoU | 0.9099 |
| Mean mask Dice | 0.9416 |
| Center x MAE | 0.500 px |
| Bottom y MAE | 14.278 px |
| Width MAE | 1.194 px |
| Height MAE | 15.514 px |

Camera poses match exactly. Maximum actor-position error is 3.81e-6 m and
maximum geometric-view error is 3.12e-5 degrees. The meaningful approach and
alongside pass are continuous and closely placed. Frames 69-71 are remaining
border-tail outliers; frame 70 scores 0.116 mask IoU and 0.196 bbox IoU.

The accepted bus replay is a calibrated hybrid. It uses the targeted
horizontal Cartesian-close
sprites where that bank covers the query and falls back to ASTRA's native-bank
projection everywhere else. Compared with native-only ASTRA, mask IoU rises
from 0.8843 to 0.9099 and angle MAE falls from 5.54 to 3.39 degrees while
visibility agreement remains 1.0000. Extending the bank through 0.25-10.5 m
removes the native-bank changes around frames 28 and 53. The late border-tail
outliers at frames 69-71 remain.

## Evidence

- CARLA: `artifacts/static_left_bus_overtake_carla_v2/carla_reference.mp4`
- Native-only baseline: `artifacts/static_left_bus_overtake_comparison_v3_carved_hull/`
- Targeted manifest: `bus_sidepass_asset_manifest_v1.json`
- Hybrid HE: `artifacts/static_left_bus_overtake_he_v8_horizontal_fullrange/he_replay.mp4`
- Hybrid pair: `artifacts/static_left_bus_overtake_comparison_v8_horizontal_fullrange/pair_comparison.mp4`
- Hybrid metrics: `artifacts/static_left_bus_overtake_comparison_v8_horizontal_fullrange/summary.json`
- Hybrid review: `artifacts/static_left_bus_overtake_comparison_v8_horizontal_fullrange/review/critical_frames_contact_sheet.png`

The earlier CARLA `v1` capture used drifted recorder defaults (900x256, FOV
100, camera x=-1.5 and z=2.0) and is superseded by camera-matched `v2`.
