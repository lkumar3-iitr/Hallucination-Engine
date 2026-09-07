# ScenarioGenerator

ScenarioGenerator converts text/user intent into executable scenarios for the Hallucination Engine (HE) and CARLA.

Current v1 implementation focus:

1. Structured Scenario Schema v1
2. Rule-Based Trajectory Generator v1
3. Resolved Scenario JSON
4. Adapter to current HE compositor

Full planned pipeline:

```text
Text / user intent
→ LLM scenario planner
→ structured scenario representation
→ BEV local map builder
→ trajectory generator
   - rule-based
   - optimization-based
   - learning-based
→ scenario validator
→ execution backends
   - HE backend
   - CARLA backend
```

Fixed HEPlacementModel v2 camera convention:

```text
image_width  = 1280
image_height = 720
fov          = 90

camera_x = 1.5
camera_y = 0.0
camera_z = 1.6

pitch = 0.0
yaw   = 0.0
roll  = 0.0
```

Camera mismatch was the main issue in earlier output. Treat this convention as fixed for v2 data generation, training, validation, and compositor integration.

## Pre-contact validation

Semantic event margins such as `ahead_by` compare actor and ego reference
points; they are not bumper-to-bumper distances. A resolved benchmark must
therefore be checked with the physical actor footprints before export.

The pre-contact re-entry example is:

```text
examples/s2_reentry_cutin_stop_precontact_1m_001_right.semantic_v2.json
```

Resolve and validate it from the `ScenarioGenerator` directory:

```powershell
python scripts/resolve_schema_v2.py examples/s2_reentry_cutin_stop_precontact_1m_001_right.semantic_v2.json --semantic --output outputs/v2_resolved/s2_reentry_cutin_stop_precontact_1m_001_right.resolved_v2.json
python scripts/validate_schema_v2.py outputs/v2_resolved/s2_reentry_cutin_stop_precontact_1m_001_right.resolved_v2.json
python scripts/validate_resolved_clearance_v1.py outputs/v2_resolved/s2_reentry_cutin_stop_precontact_1m_001_right.resolved_v2.json --asset-root <sprite-bank-root> --actor-id adv_reentry --minimum-clearance-m 1.0 --minimum-terminal-clearance-m 1.0 --output outputs/validation/s2_reentry_cutin_stop_precontact_1m_001_right.clearance_v1.json
```

`validate_resolved_clearance_v1.py` uses oriented physical rectangles and
fails if any common frame overlaps or falls below the requested clearance.
This validates the canonical resolved trajectory. Closed-loop model runs must
still record their own physical separation and termination reason because a
learned ego trajectory can depart from the canonical path.
