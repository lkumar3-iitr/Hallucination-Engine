# Scenario 8: Current Candidate Rerun

Date: 2026-09-12. Experimental ASTRA only; production unchanged.

## Setup

Ran `long_08_bus_occluded_pedestrian_001` for 801 samples (40 seconds at
20 simulation Hz), ClearNoon, NEAT front/left/right native cameras. Used
`calibrated_assets_candidate_v1.json`: bus uses the calibrated hull candidate;
pedestrian retains the previous backend. Tesla and Patrol are not adversaries
in this scenario, so this run does not validate their three-camera rendering.

The hidden pedestrian is gated by ego route progress at 55 m, with source
event start 12 s. Exact arguments and bank paths are in
`artifacts/scenario08_multiasset_he_v2/CANDIDATE_RUN.json`.
CARLA was already running; no restart or renderer edits were needed.

## Behavior

| Measurement | Earlier CARLA reference | New HE run |
| --- | ---: | ---: |
| Samples | 801 | 801 |
| Trigger frame | 201 | 197 |
| Final route progress | 200.760 m | 198.726 m |
| Final speed | 6.945 m/s | 6.553 m/s |
| Minimum logged actor clearance | 1.138 m | 1.145 m |
| Geometric overlap frames | 0 | 0 |

HE slowed and exhibited stop-and-go behavior around frames 253-336, near
the bus/pedestrian encounter. It resumed: at frame 427, progress was 92.313 m
and speed was 6.743 m/s. It did not remain stuck for the rest of the run.
This does not establish why it stopped or behavioral equivalence with CARLA.

The CARLA condition was reused from `scenario08_calibrated_carla_v1`, not
rerun today. The comparison shows independent closed loops at equal elapsed
time, not matched poses. Final progress similarity is not a renderer metric.
Zero geometric overlap is not a claim of physical collision-response parity.

## Rendering Checks

The source-selection audit covered all 2403 camera calls with zero
cross-camera source disagreements. Bus entry blend: frames 202-211;
calibrated hull: 212-385; exit blend: 386-391. No mid-pass native fallback.
Recorded coverage weights match the current guard exactly.

Reviewed comparison frames 240, 300, and 340: the bus body appears rigid in
front/left close views. Lighting/reflections and the missing cast shadow
remain visibly different. This sparse review is not exhaustive temporal QA
or a new matched-pose silhouette evaluation. Scene-depth occlusion remains
unsupported by this adapter.

## Video and Runtime

[Three-camera comparison](artifacts/scenario08_multiasset_comparison_v2/closed_loop_comparison.mp4)

Verified 801 video frames at 20 fps and decoded the final frame.
Runtime: 396.751 s wall time, 2.019 end-to-end FPS. Logged post-warmup
mean HE rendering latency is 428.207 ms; mean NEAT inference is 26.632 ms.
This is the experimental hull renderer with video/diagnostic recording and
startup work, not the earlier optimized renderer or a controlled throughput
benchmark. The 20 fps video describes simulation time, not wall-clock speed.

Evidence:

- `artifacts/scenario08_multiasset_he_v2/`: receipt, renderer metadata,
  consistency audit, CSV, runtime summary, raw videos.
- `artifacts/scenario08_multiasset_comparison_v2/`: comparison, sampled review
  frames, and paired behavior summary.

No production promotion, bank deletion, commit, or push was performed.
