# Forward pass-by V2: current renderer selector + GT-mask placement

This version fixes the main conceptual mistake in the previous preview.

## Previous preview

The previous preview did:

```text
incorrect/simple projected 3D box
        +
all 4320 candidates
        +
pick highest IoU
```

That lets the wrong sprite orientation compensate for a bad box. The selected
angle can therefore jump wildly even though the physical viewpoint changes
smoothly.

## V2

V2 deliberately separates **view selection** from **2-D placement**:

```text
actor pose + camera pose
          |
          v
CURRENT production 4320 selector
          |
          | chooses angle / distance / elevation
          | WITHOUT seeing the CARLA mask
          v
one selected sprite
          |
          v
fit only x / y / width / height to CARLA GT mask
          |
          v
natural image-boundary clipping
```

For the default preview, the final alpha is then replaced by the exact CARLA
GT mask. This produces the strongest qualitative visualization of:

> "What would the existing 4320 sprite appearance look like if our box/mask
> geometry were solved?"

This is an oracle visualization only. It is not for benchmark results.

## Files

Put these in:

```text
D:\HallucinationEngine\web
```

- `make_gtmask_renderer_copy.py`
- `render_forward_passby_gtmask_v2.py`
- `run_forward_passby_gtmask_v2.cmd`

## Important renderer snapshot behavior

`make_gtmask_renderer_copy.py` reads your CURRENT local file:

```text
D:\HallucinationEngine\driving_models\common\he_camera_renderer.py
```

and creates:

```text
D:\HallucinationEngine\web\he_camera_renderer_gtmask_oracle_v1.py
```

It also patches only the repository-root lookup because the copied file has
moved from `driving_models\common` to `web`.

The production renderer itself is not edited.

A SHA256 of the production source is printed, so we know exactly which local
renderer was tested.

## Run

```cmd
cd /d D:\HallucinationEngine\web
run_forward_passby_gtmask_v2.cmd
```

## Output

```text
D:\HallucinationEngine\web\outputs\tesla_forward_passby_gtmask_v2\
    forward_passby_gtmask_preview.mp4
    forward_passby_gtmask_side_by_side.mp4
    forward_passby_gtmask_rows.csv
    frames\
```

## Selection is not score-based

The CARLA mask does **not** participate in selecting:

- source angle
- source distance
- source elevation

Those come directly from the production view-matrix selector.

The mask-fit IoU shown in the logs is only a diagnostic for the 2-D transform
of that already-selected sprite.

## Exact-alpha preview

Default:

```text
final alpha = CARLA GT target mask
```

This is intentional for the first visual question.

If you also want to see the selected sprite's own fitted silhouette, run:

```cmd
python render_forward_passby_gtmask_v2.py ^
  --carla-dir D:\HallucinationEngine\web\carla_reference\static_left_tesla_ego_overtakes_200 ^
  --bank D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3 ^
  --output-dir D:\HallucinationEngine\web\outputs\tesla_forward_passby_gtmask_natural ^
  --frame-start 0 ^
  --frame-end 60 ^
  --natural-alpha
```
