# ASTRA Selector Optimization Results

Status date: 2026-09-08

## Frozen Baseline

The pre-optimization renderer is recorded in
`frozen/astra_passby_v1_close_hybrid_2026-09-08/CHECKPOINT.md`. Production and
targeted sprite banks were not modified.

On 27 native-bank Tesla selections, the frozen selector averaged 125.0 ms per
call. The benchmark replays recorded actor and camera poses through ASTRA's
centered virtual camera and compares selected keys with frozen frame metadata.

## Accepted Optimizations

1. Apply the existing 35-degree physical eligibility gate before silhouette
   IoU instead of scoring all 4,320 sprites and discarding ineligible results.
2. Store canonical bank masks with `numpy.packbits` and compute exact IoU using
   bytewise AND/OR plus an 8-bit popcount table.

Tesla selection now averages 35-38 ms per call, approximately 3.4 times faster
than the frozen 125.0 ms baseline. One-pass measurements on the accepted
visible native-bank poses are 35.5 ms for Tesla, 59.7 ms for Patrol, and 11.9
ms for pedestrian. Hull complexity accounts for much of the cross-asset
difference.

## Equivalence Gates

- Direct randomized tests prove packed IoU exactly equals boolean IoU for
  empty, sparse, dense, and full target masks.
- Recorded trajectory replay has zero selected-key mismatches across 27 Tesla,
  28 Patrol, and 27 pedestrian native-bank visible poses.
- A complete optimized Tesla replay has zero selected-key mismatches and zero
  SHA-256 mask mismatches across all 201 frames versus the frozen replay.

Optimized replay evidence:
`artifacts/static_left_tesla_overtake_targeted_close_optimized_v1/`

## Rejected Optimizations

- Sparse target-pixel indexing preserved IoU but slowed Tesla selection to
  73.2 ms per call because NumPy advanced indexing dominated.
- Replacing per-face `fillConvexPoly` with batched `fillPoly` was
  performance-neutral and changed two Patrol and six pedestrian selections.
  The frozen per-face rasterization was restored.

## Remaining Cost

The main remaining selector costs are exact hull rasterization and exact packed
IoU. End-to-end replay also includes synchronous CARLA ticks, sprite loading,
compositing, and video encoding, so selector acceleration does not map directly
to the same whole-run speedup.
