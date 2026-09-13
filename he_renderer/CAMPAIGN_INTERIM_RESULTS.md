# Runtime Campaign Interim Results

2026-09-13. Source: runtime_campaign_v1 PLAN.json, completed runtime summaries
and frame CSVs. Campaign stopped after 45/168 jobs: eight smoke jobs and 37
coverage jobs. One repetition, development scenarios, incomplete coverage.

## Completed Pairs

Final route progress in metres, CARLA / HE. These are independently controlled
trajectories, not matched-pose image quality or statistical equivalence results.

| Scenario | NEAT | TCP | CIL++ | AIM-MT |
| --- | ---: | ---: | ---: | ---: |
| 01 | 178.0 / 178.9 | 175.0 / 162.5 | 189.8 / 191.6 | 176.4 / 177.2 |
| 02 | 74.1 / 73.6 | 73.0 / 72.5 | 73.7 / 74.1 | 72.7 / 73.0 |
| 03 | 84.6 / 84.9 | 83.8 / 84.2 | 85.6 / 85.3 | 83.5 / 83.7 |
| 04 | 82.9 / 81.1 | 80.3 / 79.7 | 81.9 / 81.5 | 79.9 / 80.1 |
| 05 (partial) | 200.5 / 198.0 | 169.1 / 69.2 | 209.4 / incomplete | pending |

No logged physical_overlap frames in either condition for scenarios 01-04.
Scenario05 NEAT has 11 CARLA overlap frames versus zero HE; CIL++ CARLA has
17 overlap frames. Counts are geometric overlap frames, not collision events
or collision-sensor readings. The failed CIL++ HE run is not a zero-overlap
completed result. TCP Scenario05 has no overlaps in either completed run but
a large progress difference. Neither outcome should be discarded.

For scenarios 01-04, live HE mean render latency ranges from 6.5 to 21.4 ms
per camera set; model-inclusive throughput ranges from 14.8 to 41.8 FPS.
These measurements include shared-GPU CARLA contention and differ from offline
renderer throughput. Scenario02-04 progress agreement is encouraging, but
endpoint progress alone does not establish matching braking or safety behavior.

## Failure Diagnosis

Job045 fails in GPUNativeWarp -> aimed_camera. With physical extents supplied,
the aiming target is the nearest point on the physical bounding box. The
"Camera is at the actor center" exception actually tests distance to that
target below 0.05 m. It therefore does not prove coincidence with actor center.
The selector independently rejects hull vertices at or behind its 0.01 m
near plane. Simply removing the distance check is not a validated solution.

Next gate: reproduce the failed camera and actor poses, distinguish external
near-surface geometry from camera-inside-box geometry, and define/test the
rendering domain at contact. Do not hide actors, suppress arbitrary exceptions,
or count a truncated run as successful. Keep the campaign stopped until that
gate passes. Any changed renderer needs a new source receipt and explicit
campaign revision; retain all existing results and the failed attempt.
