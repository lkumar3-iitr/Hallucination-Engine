# Contact Geometry Diagnosis

## Terminal Policy Implemented

The user approved termination at first collision. The common runner now stops
both conditions at the first scenario-actor 2-D footprint overlap, before
rendering or inference. This is the existing geometric metric, not a CARLA
collision-sensor event, a swept continuous collision test, or detection of
collisions with all static scenery. Detection is sampled at simulation ticks.

`collision_event.json` and the runtime summary record the terminal frame,
simulation time, actor ID, ego pose and criterion. Frame CSVs and videos contain
only processed pre-contact frames; consumers MUST read the terminal receipt
to classify collision outcomes, not infer no collision from those CSV rows.
The CLI defaults to `--stop-on-collision`; `--no-stop-on-collision` retains the
old diagnostic policy but is not accepted by the revised campaign.

Live CIL++ Scenario05 gates: contact_terminal_he_v1 and
contact_terminal_carla_v1 both terminate at frame233, 11.65s, with 233 processed
frames and clean exit. Both summaries pass the campaign terminal validator.
The last HE mosaic was inspected and contains the approaching Nissan; no
post-contact frame was fabricated. These independent runs do not prove general
behavioral equivalence. All 45 unit tests pass.

The revised campaign uses a new output root, runtime_campaign_terminal_v2,
with 16 contact/integration smoke jobs followed by 160 coverage jobs. Prior
campaign data are retained and are not pooled silently with this new protocol.
Asset payloads and rendering algorithms are unchanged by the terminal policy.

## Original Investigation

2026-09-13. The original campaign and failed outputs remain untouched.

An independent CIL++ Scenario05 rerun reproduces the same unsupported contact
domain, this time at the inside-bbox guard rather than the 5 cm surface guard.
Evidence: artifacts/contact_diagnosis_v1/failure_pose.json and run/.
The rerun has a different independently controlled trajectory; it is not an
exact replay of the original failed frame.

Recorded actor-local camera: (-2.3586343, 1.0689399, 1.9784338) m.
Physical box center: (0.0283829, 0.00000225, 1.0167187) m.
Half-extents: (2.7829144, 1.0749835, 1.0225736) m.
All three coordinates lie inside the bounds: nearest-box clearance is zero.
The original campaign's last completed frame (243) already records
physical_overlap=1. This is not a camera literally coinciding with actor center.

## Change and Remaining Decision

The aiming helper now raises a ValueError subclass with an explicit reason and
target distance. Supported rendering behavior and rejection thresholds are
unchanged. Tests cover the captured inside pose, a 2 cm external surface pose,
and a supported external pose. This improves diagnosis; it does not implement
post-contact rendering or resolve the campaign blocker by itself.

Two possible evaluation contracts need to be distinguished:

- End both conditions at first geometric overlap, retain the collision outcome
  and time to contact, and treat frame counts as valid terminal lengths. This
  avoids comparing CARLA impact dynamics with virtual interpenetration. It
  requires checking contact before rendering and updating completeness gates.
- Continue after overlap. This requires a separately validated rendering method
  for near-plane and inside-actor views. Removing guards, hiding the actor or
  freezing a previous image would not establish correct rendering.

At the time of the original diagnosis no contact policy had been changed.
The approved implementation and its evidence are recorded above.
