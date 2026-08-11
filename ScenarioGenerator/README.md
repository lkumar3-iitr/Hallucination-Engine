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
