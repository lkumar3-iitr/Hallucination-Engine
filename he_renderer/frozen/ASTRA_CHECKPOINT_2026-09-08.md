# ASTRA Close-Hybrid Freeze Checkpoint

Frozen: 2026-09-08

Purpose: preserve the accepted Tesla and Patrol side-pass implementation and
evidence before pedestrian recorder repair and selector optimization.

## Accepted Evidence

| Asset | Visible | Visibility | Mask IoU | Dice | Bbox IoU | Bottom MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesla Model 3 | 61 | 1.0000 | 0.9429 | 0.9658 | 0.9564 | 1.672 px |
| Nissan Patrol | 62 | 1.0000 | 0.9464 | 0.9722 | 0.9710 | 1.823 px |

Evidence directories:

- `astra/artifacts/static_left_tesla_overtake_targeted_close_v1_comparison/`
- `astra/artifacts/static_left_patrol_overtake_targeted_close_v1_comparison/`

Both runs use `cartesian_close` for 34 close-pass frames and the production
4320-view `view_matrix` bank for all other frames.

## Repository File Hashes

SHA-256:

```text
astra/passby_renderer.py 7223a0729adf82ade82771247987d87afe3062dc7703e583b877ccced2aced4b
astra/selector.py 56abb663c8255fef8d9b54bbbf90bedfd6069844fc55270c3c19786e57e9d649
astra/compositor.py 4e4b7ecdd8fdb7b32223c4ba1295cfab03671ce3d5c01da8dd1469dd1bc58cfa
astra/run_pair.py 5438f545ddeace5a83a7afb5a973b7194c65adb215df942f9f62ef536015cb73
driving_models/common/record_scenario_carla_he_pair_v3.py 28a8fd91077fc4e3ba4e05f15e8c1605d23e6f75cd730427d9d5bb89e04d7937
astra/tesla_sidepass_asset_manifest_v1.json 1f8ac76c4f8eab093dfb1df3dca750b95cb1fba1141269c14981e235533ba1f4
astra/patrol_sidepass_asset_manifest_v1.json 675f7e8c26324017ee07dd86b0ae6a3817f2195ad8b685e30d8dd6fd98d4c6cb
astra/pedestrian_sidepass_asset_manifest_v1.json 0831135b0ebb5e852c09b203f2ed5455ec9d91e8495c9506482ed2ef7058f07a
```

## External Bank Manifest Hashes

These hashes cover each bank's `view_matrix.csv`; RGBA assets remain in the
external asset repository.

```text
tesla_model3_sidepass_level_left_v1 02ad0109f30376384cd328da74c05ba3b9237e2049037b141256a033bf6f6cee
tesla_model3_sidepass_level_right_v1 791a73345b657c410f6918e0018bc502448fc4675e492b8dfe11f88f27beb2a4
nissan_patrol_2021_sidepass_level_left_v1 ecb40bd2b81cbaaa30a6e81fe3b07e4889902e1d3c6add63a34418b002a7711e
nissan_patrol_2021_sidepass_level_right_v1 3e89ab563bc0e57cef149372daaeef250ca8a7f15ebf9a4736cede37bc994f7e
pedestrian_0001_sidepass_level_left_v2 0399ec576af6c76e88a34157eebef4983788433d07e4594beda247cd963fcbaa
pedestrian_0001_sidepass_level_right_v2 50918a6f37695740a150d5725b63591f6892d884b5709b99f4cb3993905d184f
```

## Optimization Acceptance Rule

An optimized selector must first reproduce this checkpoint's per-frame
selected sprite keys and output masks on Tesla and Patrol. Any intentional
selection change requires a new pair comparison with no visibility regression
and metrics at least as strong as the table above.

The production 4320 banks and targeted close banks must not be modified during
optimization.
