# Selector Optimization Results

Status date: 2026-09-09

The exact CPU renderer was first reduced from 116.07 ms to 32.46 ms per
1280x720 native-bank render by limiting inverse reprojection to its analytical
visible ROI, caching decoded sprites, and removing redundant packed-mask IoU
work. These changes preserved every selected key across 576 direct comparison
poses and preserved RGB and alpha exactly on the accepted native trajectory.

A 100,000-pose teacher table was then generated from the exact visual-hull
selector over bearing, log-distance, and elevation. Runtime nearest-neighbor
selection uses a neighbor-silhouette confidence gate of 0.935 and falls back
to the exact selector when uncertain.

On an independent 5,000-pose masked validation set, the unguarded table had
0.9300 mean canonical silhouette IoU versus 0.9317 for the exact teacher. Its
mean regret was 0.0017, and 98.72% of poses were within 0.02 of the teacher.

Across 576 preserved trajectory poses with the 0.935 guard:

| Metric | Result |
| --- | ---: |
| Distilled fraction | 65.10% |
| Mean mask IoU vs exact, all poses | 0.9970 |
| Mean mask IoU vs exact, visible poses | 0.9927 |
| Minimum visible mask IoU vs exact | 0.9024 |
| Visible poses at or above 0.95 IoU | 99.16% |

On the accepted Tesla native trajectory, the guarded renderer used distilled
selection on 85.19% of poses and measured 8.28 ms per render, or 120.74 FPS,
after warm-up. This is an offline single-actor, single-camera CPU measurement;
it excludes CARLA synchronization and video encoding.

Rejected optimizations included 64x64 selector masks, batched polygon filling,
front-face-only voxel rasterization, a narrowed angle window, raw tree
ensembles, and a small MLP. Each either changed validated selections, produced
poor uncertainty behavior, or failed the latency target.
