# Rendering Method

## Inputs

For each actor and camera frame, the renderer consumes the actor transform,
camera transform, image dimensions, horizontal field of view, and an asset
manifest. A manifest points to a native view-matrix bank and may also register
an asset-specific Cartesian close bank. The runtime does not consume target
CARLA pixels, masks, or bounding boxes.

## Native Bank Path

1. Reconstruct a coarse visual hull from the alpha masks and calibrated camera
   metadata in the existing sprite bank.
2. Place a virtual camera at the native camera origin and aim it toward the
   nearest valid point in the actor bounding box.
3. Project the visual hull into that centered camera and rank eligible bank
   sprites by normalized silhouette overlap. Angle filtering and packed-mask
   popcount reduce candidate-scoring cost without changing the selected key.
4. Compute the full, unclipped sprite placement in the centered camera.
5. Map native image rays into the centered camera with the pure rotational
   homography induced by the two cameras sharing the same optical center.
6. Inverse-sample the premultiplied RGBA sprite, clip at the viewport, and
   reject rays that point behind the centered camera.

This construction preserves visible side fragments while an extended actor
crosses the camera plane, even when its center is no longer in front.

## Close Bank Path

At near-field side-pass poses covered by a registered Cartesian bank, the
compositor uses the calibrated close-bank renderer. These banks place the
asset at controlled forward and lateral offsets and exclude camera-geometry
intersections. They provide near-horizontal side views that are undersampled
or distorted in the original spherical 4320-view captures. Outside their
coverage, selection returns to the native-bank path.

## Visibility And Composition

Actors are sorted far-to-near for composition. Alpha is sampled and blended
in premultiplied form. Visibility follows valid projected rays and alpha
support, rather than an actor-center depth gate. Scene-depth occlusion is
deliberately rejected until a depth-aware implementation is validated.

## Reproducibility

Every experiment should record `he_renderer_backend=he_sprite_renderer_v1`
and each actor's `sprite_mode` (`cartesian_close` or `view_matrix`). The frozen
checkpoint records the hashes of the accepted pre-rename implementation.
