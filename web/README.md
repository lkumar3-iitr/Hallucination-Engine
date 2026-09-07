# HE 4320 Full-Bank Silhouette Oracle

This folder tests one question only:

> If the CARLA-visible 2D box is already perfect, does the original 4320 Tesla sprite bank contain a view whose silhouette matches CARLA closely, including close range?

The experiment intentionally ignores the current HE angle formula, distance interpolation, close-range Cartesian bank, and learned placement logic.

## Inputs

### 1. CARLA reference run

Use a directory produced by:

`driving_models\common\record_scenario_carla_he_pair_v3.py`

It should contain:

```text
setup.json
frames.jsonl
masks\
carla_reference.mp4
```

The search uses only the CARLA condition. An HE replay is not required.

Optionally pass:

```text
--he-dir <matching HE replay directory>
```

to log the current HE requested/selected angle beside the oracle winner. This is diagnostic only and never constrains the search.

### 2. Original 4320 bank

The bank directory should contain:

```text
view_matrix.csv
...
RGBA PNG sprites
```

The expected grid is:

- 360 azimuth angles
- 4 source distances: 5, 10, 15, 20 m
- 3 source elevations: 0, 5, 10 deg

Total: `360 x 4 x 3 = 4320`.

The script prints the grid it actually finds. With `--strict-4320`, it stops if the usable row count is not exactly 4320.

## Why distance is normalized

The source sprite alpha is first cropped to its own visible alpha bounds, then resized to a canonical square for the coarse search.

The CARLA target mask is also cropped to its exact visible CARLA box and resized to the same canonical square.

Therefore a 5 m bank image does not get an advantage simply because it contains more pixels than a 20 m image.

During native refinement, the candidate alpha is resized directly into the exact CARLA GT box. This is intentional: this first experiment assumes that the box problem has already been solved.

## Why elevation is NOT normalized away

Elevation changes which roof/hood/windows/sides are visible. That is real viewpoint information.

All 0/5/10 degree source elevations are allowed to compete. The winning elevation is written to the CSV.

This is especially useful for later camera-general HE:

```text
target camera pose + actor pose + FOV
             |
             v
 target distance / bearing / elevation
             |
             v
     sprite selector or learned mapping
```

## FOV

The target FOV is read from `setup.json` and saved in every result.

For this exact-box oracle, FOV magnification is already removed by GT-box normalization. If the bank still cannot match the silhouette after that, the error is not simply "wrong pixel scale."

Later, when we stop using GT boxes, the target camera intrinsics/FOV become inputs to placement and projection.

## Camera pose

For every frame the output records:

- camera pitch/yaw/roll
- `camera_forward_m`
- `camera_right_m`
- `camera_up_m`
- camera-to-actor center distance
- camera-to-actor horizontal distance
- camera-position elevation angle
- CARLA geometric view angle
- current HE-selected angle, if present in the CARLA metadata
- oracle bank angle/distance/elevation

This gives us the training/calibration table for the next stage if the oracle succeeds.

## Metrics

Primary ranking:

**symmetric boundary Chamfer distance**

Lower is better.

Also reported:

- mask IoU
- Dice
- boundary F1 at 1 px
- boundary F1 at 2 px
- boundary F1 at 3 px

IoU is deliberately secondary.

## First run: smoke test

Edit paths in:

`run_oracle_smoke.cmd`

Then:

```cmd
cd /d D:\HallucinationEngine\web
run_oracle_smoke.cmd
```

This tests nine frames across the pass.

Inspect:

```text
outputs\tesla_oracle_smoke\oracle_summary.json
outputs\tesla_oracle_smoke\oracle_results.csv
outputs\tesla_oracle_smoke\diagnostics\
```

## Full 200-frame run

Edit paths in:

`run_oracle_full.cmd`

Then:

```cmd
cd /d D:\HallucinationEngine\web
run_oracle_full.cmd
```

## Outputs

```text
oracle_results.csv
oracle_summary.json
oracle_diagnostic.mp4

best_masks\
    frame_000020.png
    ...

top_candidates\
    frame_000020.csv
    ...

diagnostics\
    frame_000020.png
    ...
```

## What would support the 4320-bank hypothesis?

The most important plot/table later will be error versus true camera-to-actor distance.

A strong result would look like:

- boundary error stays low from far range through the side-by-side close pass;
- no sharp failure at 5 m, 4 m, 3 m;
- oracle source angle evolves smoothly or at least predictably;
- one of the existing 0/5/10 elevation views remains adequate;
- IoU/Dice stay high even when the object is partially clipped by image borders.

If the oracle itself fails badly at close range even with the perfect CARLA box, then the original 4320 bank is genuinely missing required viewpoints/appearance.

If the oracle works but the current HE render fails, the bank is not the problem: selection/placement is.
