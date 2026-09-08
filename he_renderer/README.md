# HE Sprite Renderer

`he_sprite_renderer_v1` is the frozen production renderer for the current
Hallucination Engine experiments. It composites actors from pre-generated
CARLA sprite banks using only actor pose, camera pose, camera intrinsics, bank
metadata, and sprite alpha masks. Runtime CARLA target RGB, instance masks,
and target boxes are not renderer inputs.

## Layout

- `renderer.py`: centered-camera selection, placement, and native-camera reprojection.
- `selector.py`: bank-only visual-hull selector with packed-mask scoring.
- `compositor.py`: adapter for the existing multi-actor recorder contract.
- `manifests/`: accepted native and close-bank registrations.
- `scenarios/`: reproducible overtake and 40-second NEAT inputs.
- `runners/`: pair-recording and scenario-construction entry points.
- `evaluation/`: silhouette, bbox, visibility, and artifact checks.
- `tools/`: capture, preview, selector benchmark, and video utilities.
- `tests/`: geometry and optimized-selector regression tests.
- `frozen/`: checksums and evidence paths from the pre-rename checkpoint.

Generated videos, frames, masks, and selector caches are intentionally not
tracked. Store them outside this package or in ignored output directories.

## Validate

From the repository root:

```powershell
python -m unittest discover -s he_renderer/tests -p "test_*.py"
```

## Record A Pair

CARLA must already be running. Record the physical condition with the existing
recorder, then use the production renderer for the matched HE condition:

```powershell
python he_renderer/runners/run_pair.py --condition he `
  --resolved he_renderer/scenarios/static_left_tesla_ego_overtakes_200.resolved_v2.json `
  --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets/sprite_bank_native_production `
  --manifest he_renderer/manifests/tesla_sidepass_asset_manifest_v1.json `
  --reference-dir web/carla_reference/static_left_tesla_ego_overtakes_200 `
  --output-dir outputs/he_sprite_renderer_v1/tesla_overtake `
  --actor-id left_tesla --max-frames 201
```

Use `--he-renderer-version he_sprite_renderer_v1` with the generic closed-loop
runner. Select the manifest matching the scenario asset. `cartesian_close` is
selected when its registered close bank covers the pose; otherwise the
renderer uses the 4320-view `view_matrix` bank.

See `CURRENT_STATE.md` for accepted measurements and `METHODS.md` for the
rendering procedure and its current limitations.
