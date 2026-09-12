# Compact Multi-Asset Candidate Results

Date: 2026-09-12. All changes and generated outputs remain inside ASTRA.
No production files, native banks, or old close banks were replaced or deleted.

## Storage and Capture Decision

The existing targeted banks have complete calibration but incomplete ray
coverage for this renderer: Tesla/Patrol forward offsets are 0.25-10.5 m,
with one height and many clipped views; pedestrian begins at 2.25 m, also at
one height. They cannot provide the full approach/alongside/departure rays.
They remain available to the previous renderer; they were not duplicated.

New compact banks use 21 forward nodes (-10..10 m, 1 m spacing), four signed
lateral nodes (-4.5, -3, +3, +4.5 m), and two vertical levels: 168 views each.
Captures use 1536x1536, FOV120, level cameras with horizontal angular-midpoint
aiming. Only tightly cropped RGBA images and calibration are stored.

| Asset | Complete bank | Views | Size including metadata |
| --- | --- | ---: | ---: |
| Tesla | `artifacts/tesla_aimed_close_compact_v2` | 168 | 21.51 MiB |
| Patrol | `artifacts/patrol_aimed_close_compact_v1` | 168 | 31.83 MiB |
| Pedestrian | `artifacts/pedestrian_aimed_close_compact_v1` | 168 | 1.94 MiB |
| Total | | 504 | 55.28 MiB |

This is additional close-bank storage, not the total project size. Existing
native banks remain dependencies. Comparison videos, sparse review frames,
and derived caches are separate. A 4.94 MiB partial Tesla v1 bank is preserved
but unused; no cleanup was performed. The active bus close bank is unchanged.

Capture QA rejected the first Tesla configuration because of vertical
clipping. Increasing canvas height preserved the focal length and captured
the complete vehicle without storing the empty canvas. Target identification
now uses the actor's projected-center neighborhood and a pinned instance key:
static map vehicles can share semantic tag 14 even on an idle CARLA server.
Ambiguous and clipped captures fail rather than enter the completed bank.

## Matched Overtake Tests

Each asset was replayed for all 201 frames using its existing frozen CARLA
reference and pose-matched clean background. No inpainting or new closed-loop
trajectory was used. Calibration, frame IDs/counts, actor transforms, and
camera transforms were checked before rendering. Evaluation uses actual
compositing alpha > 10/255 against CARLA masks, including tiny border frames.

| Asset / candidate | Mean visible mask IoU | Earlier accepted hybrid | Dropouts | Ghosts |
| --- | ---: | ---: | ---: | ---: |
| Tesla, native-refined hull | 0.9494 | 0.9429 | 0 | 0 |
| Patrol, native-refined hull | 0.9488 | 0.9464 | 0 | 0 |
| Pedestrian, native-refined hull | 0.7692 | 0.8060 | 1 | 0 |
| Pedestrian, close-only 1 cm hull diagnostic | 0.7710 | 0.8060 | 1 | 0 |
| Pedestrian, center-plane diagnostic | 0.7458 | 0.8060 | 0 | 0 |

Visible GT frames: Tesla 61, Patrol 62, pedestrian 52. The pedestrian dropout
is frame 51 with 137 GT pixels and no predicted pixels. It is not omitted.

Tesla frame 50 reaches 0.9896 IoU; Patrol frame 50 reaches 0.9828. Reviewed
close frames show rigid vehicle bodies, with the existing appearance/color
and illumination differences. Improvements in full-sequence IoU are modest;
these tests do not establish generalization to every camera or interaction.

The pedestrian candidate is **not accepted**. Its rendered arm/foot contours
are deficient. A wider, finer hull carved from the new masks barely improves
the result, so coarse voxel spacing or the narrow initial hull alone does not
explain the failure. The plane diagnostic also fails. Pose and mask consistency
need a separate audit; no claim of solved pedestrian geometry is made.

## Videos

Follow-up: [Scenario 8 NEAT three-camera rerun](SCENARIO_08_MULTI_ASSET_RERUN.md)
completed with the current manifest. It exercises bus plus the retained
pedestrian backend, not Tesla or Patrol.

- [Tesla](artifacts/tesla_calibrated_compact_passby_v1/comparison.mp4)
- [Patrol](artifacts/patrol_calibrated_compact_passby_v1/comparison.mp4)
- [Pedestrian, rejected candidate](artifacts/pedestrian_calibrated_compact_passby_v1/comparison.mp4)
- [Pedestrian, fine-hull diagnostic](artifacts/pedestrian_calibrated_compact_fine_v2/comparison.mp4)

Every evaluation directory includes `summary.json`, `rows.csv`, and source
metadata. Review JPEGs are saved every ten frames to limit redundant storage.

## Integration and Preservation

`calibrated_asset_renderer.py` exposes the same rendering core for each asset
and rejects native/close blueprint mismatches. The bus rendering core remains
unchanged. `calibrated_compositor.py` accepts explicit per-blueprint banks;
without the new manifest it retains the prior bus-only opt-in behavior.

`calibrated_assets_candidate_v1.json` opts bus, Tesla, and Patrol into the
experimental path. Its pedestrian entry is disabled, so that actor retains
the previous backend. This is not a production manifest or approval.

To use it with the isolated runner, add:

```powershell
--candidate-manifest astra/calibrated_assets_candidate_v1.json
```

The complete multi-actor closed-loop runner was not rerun for the new assets
in this test. These results are forward-camera, static-actor overtake replays;
mirrored passes, NEAT three-camera tests for the new assets, and animated
pedestrian poses remain unvalidated.

Pre-change bus code snapshot:
`frozen/bus_candidate_before_multiasset_20260912.zip`.

The optional finer hull has a separate cache identity, including bounds and
spacing. Existing caches and accepted evidence remain intact. All 22 ASTRA
unit tests pass, including blueprint binding and disabled pedestrian gating;
modified modules compile. CARLA capture completed without requiring a restart.

## Reproduce

From the repository root in the `he_neat` environment, choose a new output:

```powershell
python -m astra.evaluate_calibrated_close --reference web/carla_reference/static_left_tesla_ego_overtakes_200 --background astra/artifacts/static_left_tesla_overtake_targeted_close_v1 --close-bank astra/artifacts/tesla_aimed_close_compact_v2 --native-bank D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production/tesla_model3_native_full_v3 --output astra/artifacts/my_tesla_compact_test --hull --refine-hull --save-every 10
```

Each bank's `config.json` records its capture grid and actor blueprint. Bank
preparation uses all close masks; runtime sampling blends at most eight
neighbors, not all 168 images. Unused views in this one trajectory must not be
deleted blindly: other poses and hull reconstruction can need them.
