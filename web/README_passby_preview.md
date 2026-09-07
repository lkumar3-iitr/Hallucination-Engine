# Forward pass-by preview with clipping

This folder now includes a **qualitative preview renderer** for the original 4320 Tesla bank.

## Purpose

Before changing the HE runtime, generate a pass-by video for the **forward camera** and visually inspect:

- whether the 4320 bank already contains convincing close side appearances,
- how natural clipping at image boundaries looks,
- how the selected source angle / distance / elevation evolve over time.

This renderer is deliberately **offline** and uses an existing CARLA reference recording.

## Inputs

### CARLA reference run

Use:

```text
D:\HallucinationEngine\web\carla_reference\static_left_tesla_ego_overtakes_200
```

Expected contents:

```text
setup.json
frames.jsonl
masks\
carla_reference.mp4
```

### Original 4320 Tesla bank

Use the bank root:

```text
D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3
```

## How it works

For each visible frame:

1. Read the CARLA RGB image and CARLA instance mask.
2. Compute a **full projected 2D box** from the actor 3D dimensions and the recorded camera/actor transforms.
3. Try every sprite in the 4320 bank.
4. Resize the sprite into that full projected box.
5. Let image clipping happen naturally at frame boundaries.
6. Compare the resulting visible silhouette against the CARLA mask.
7. Keep the best sprite.
8. Remove the physical Tesla approximately from the CARLA RGB frame using inpainting.
9. Composite the selected sprite back onto that background.

So this is a **"what would a clipping-aware 4320-bank preview look like?"** tool.

## Run

From CMD:

```cmd
cd /d D:\HallucinationEngine\web
run_forward_passby_preview.cmd
```

## Outputs

```text
outputs\tesla_forward_passby_preview\
    oracle_passby_preview.mp4
    oracle_passby_side_by_side.mp4
    oracle_passby_rows.csv
    frames_preview\
```

### Videos

- `oracle_passby_preview.mp4`
  - just the preview composite

- `oracle_passby_side_by_side.mp4`
  - top-left: CARLA RGB with physical Tesla
  - top-right: preview composite
  - bottom-left: CARLA target mask
  - bottom-right: absolute RGB difference

## Important note

This is still **not** the final HE renderer. It is a visualization / probing tool.

The 2D placement box currently comes from simple projected physical dimensions, not the final learned/validated HE box model. So if the preview is promising, the next step is to keep the 4320-bank selection logic and replace only the box/placement part with the proper HE geometry pipeline.
