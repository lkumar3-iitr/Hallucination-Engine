# Fixed GT-mask pass-by package

This package replaces the broken V2 package.

The earlier `make_gtmask_renderer_copy.py` was malformed at:

```python
ADDON = r
```

That was a file-generation error.

This corrected version is simpler and safer:

1. `make_gtmask_renderer_copy.py`
   - copies the CURRENT local production renderer;
   - patches only its repo-root lookup because it moved into `web`;
   - does not embed or append a giant string;
   - does not change production renderer logic.

2. `gtmask_oracle_helpers.py`
   - contains the experiment-only GT-mask fitting code.

3. `render_forward_passby_gtmask_v2.py`
   - imports the copied renderer;
   - calls the production `load_view_matrix_sprite_bank`;
   - calls the production `select_view_matrix_sprite`;
   - uses CARLA GT only to fit x/y/width/height after selection.

## Run

Overwrite the previous three experiment files in:

```text
D:\HallucinationEngine\web
```

with this package, then:

```cmd
cd /d D:\HallucinationEngine\web
run_forward_passby_gtmask_v2.cmd
```

## Expected first-stage output

You should first see:

```text
CURRENT HE RENDERER SNAPSHOT CREATED
[source]        D:\HallucinationEngine\driving_models\common\he_camera_renderer.py
[source sha256] ...
[output]        D:\HallucinationEngine\web\he_camera_renderer_gtmask_oracle_v1.py
[production]    unchanged
```

Then the production view-matrix loader should print:

```text
[ViewMatrix] Loaded 4320 views ...
[ViewMatrix] Angles: 360
[ViewMatrix] Distances: [5.0, 10.0, 20.0]
[ViewMatrix] Elevations: [0.0, 5.0, 10.0, 20.0]
```

## Critical interpretation

`fitIoU` is allowed to use the GT mask because it optimizes only:

```text
center_x
bottom_y
width
height
```

It cannot change:

```text
selected angle
selected distance
selected elevation
```

Those three come from the production selector before GT fitting starts.
