# Hallucination Engine Current State

Status date: 2026-09-08

Accepted renderer candidate: `astra_passby_v1`

## What Works

HE can render a virtual Tesla Model 3 from the existing 4320-sprite bank into
a CARLA camera sequence without a physical adversary in the background video.
The current ASTRA path uses actor pose, camera pose, camera intrinsics, bank
metadata, and bank alpha masks. Runtime CARLA RGB, instance masks, and target
boxes are not renderer inputs.

The renderer now handles the failure that motivated this work: an ego vehicle
approaching, drawing alongside, and overtaking a static HE adversary. It keeps
the visible fragment while the actor spans the camera plane and removes it
after all valid rays leave the viewport. It does not use actor-center depth as
a whole-object visibility decision.

The existing 4320 bank is sufficient for the tested Tesla viewpoints. The
ASTRA selector reconstructs a coarse bank-only visual hull and uses its
runtime projection to rank the existing sprites. On selector-only tests it is
within 0.02 normalized silhouette IoU of exhaustive 4320-sprite search on all
138 evaluated poses.

## Accepted Scenario Result

Scenario:
`static_left_tesla_ego_overtakes_200.resolved_v2.json`

The complete 201-frame HE replay has exact camera transforms relative to the
frozen CARLA reference and 100% visibility agreement. Across 61 visible frames:

| Metric | Result |
| --- | ---: |
| Mean mask IoU | 0.9291 |
| Mean mask Dice | 0.9562 |
| Mean bbox IoU | 0.9424 |
| Center x MAE | 1.041 px |
| Bottom y MAE | 3.787 px |
| Width MAE | 3.426 px |
| Height MAE | 4.344 px |

The side-pass is visually continuous through frames 45-59. Frame 60 remains a
border-tail outlier: CARLA contains 38 pixels and ASTRA contains 1099 pixels.
Frames 61-200 are correctly empty.

Independent 125-frame sweeps at native camera yaws 0, -60, and -90 degrees
have zero ASTRA dropouts and zero post-exit ghosts. Visible-frame mean IoUs are
0.9397, 0.9414, and 0.9452. Directly alongside frames 52-72 in the -90-degree
sweep average 0.9607 IoU.

## Renderer Architecture

1. The sprite bank supplies appearance, exact capture calibration, alpha masks,
   and physical bounding-box metadata.
2. A bank-only visual hull predicts the vehicle outline for a requested pose.
3. The selector ranks existing sprites against that predicted outline.
4. A virtual camera at the native camera origin is aimed at the actor center.
   Selection and full unclipped placement occur in this stable camera.
5. A pure camera rotation maps the selected sprite into the native image.
   Inverse sampling performs viewport clipping and discards invalid rear rays.

`astra/compositor.py` adapts this path to the existing multi-actor result
contract. `astra/run_pair.py` injects it into the existing pair recorder for
one process. The accepted production candidate is therefore available and
reproducible, but it is not yet the repository-wide default backend.

## Version Decision

Use `astra_passby_v1` as the final renderer version for the current Tesla
paper experiments. No additional temporal smoothing is applied. In the
accepted scenario, selected bearing progresses monotonically with viewpoint;
the visible sequence has bank-distance transitions at frames 20 and 39 and an
elevation transition at frame 48. Adding history-dependent hysteresis now
would risk lag during the rapid close pass without evidence of a visual gain.

Always record the renderer identity as `astra_passby_v1` in experiment setup
and per-actor metadata. Keep the old renderer available as a baseline.

## Remaining Limits

- RGB illumination and reflections differ because sprites preserve source
  capture appearance.
- The final few border pixels are less accurate than the main visible pass.
- Scene-depth occlusion is explicitly unsupported by the ASTRA adapter.
- Multi-actor overlap and non-Tesla banks have not been validated.
- The current selector is not optimized for real-time throughput.
- Contact/intersection with the virtual vehicle is outside the validated domain.

These limits do not invalidate the static-adversary overtake result, but they
must remain explicit in paper claims. Current evidence supports simulator
rendering equivalence for the tested geometry and scenario, not general visual
or behavioral equivalence across all assets, cameras, and interactions.

## Reproduction

Run the test suite:

```powershell
python -m unittest discover -s astra -p test_*.py
```

Run the accepted scenario against an existing CARLA reference:

```powershell
python astra/run_pair.py --condition he --resolved important_curated_scenarios/resolved/static_left_tesla_ego_overtakes_200.resolved_v2.json --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production --reference-dir web/carla_reference/static_left_tesla_ego_overtakes_200 --output-dir astra/artifacts/my_final_replay --actor-id left_tesla --max-frames 201
```

Generated evidence stays under `astra/artifacts/` and is intentionally not
tracked by Git. See `SIDE_PASS_RESULTS.md` for detailed evidence paths.
