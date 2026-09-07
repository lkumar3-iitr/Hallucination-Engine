# Important Curated Scenarios

This folder contains high-priority ScenarioGenerator scenarios before they are
promoted into the frozen paper suite.

The current scenario is authored as semantic ScenarioGenerator v2 input, then
compiled and resolved with:

```powershell
python ScenarioGenerator\scripts\resolve_schema_v2.py `
  important_curated_scenarios\semantic\<scenario>.semantic_v2.json `
  --semantic `
  --output important_curated_scenarios\resolved\<scenario>.resolved_v2.json
```

## Scenario: Left Cut-In, Stop, Resume, Left Stop

Intent:

- straight-road route
- Tesla adversary starts in the left adjacent lane
- adversary cuts into the ego lane
- adversary brakes firmly to a full stop
- ego model should brake
- adversary waits, accelerates, moves back to the left adjacent lane, and stops
- ego model should resume once the ego lane clears

Weather variants:

- `left_cutin_stop_resume_left_stop_clear`: `ClearNoon`
- `left_cutin_stop_resume_left_stop_hardrain`: `HardRainNoon`

The two variants share identical semantic motion. The weather difference is
kept in environment sidecars so behavior, route, and geometry remain comparable.

## Renderer Calibration: Static Left Tesla Overtake

`static_left_tesla_overtake_clear_long600` keeps a Tesla stationary and
parallel in the left adjacent lane for 600 frames. It is intentionally separate
from the behavioral cut-in benchmark: the ego approach and pass expose the
front, close-side, and rear-view bank transitions without adversary motion.

The shorter renderer acceptance pair records 200 frames per run:

- `static_left_tesla_ego_overtakes_200`: ego passes a stationary left-lane Tesla.
- `moving_left_tesla_overtakes_static_ego_200`: Tesla passes a stationary ego.

Both use the same 6 m/s relative speed, 20 m initial longitudinal separation,
and 3.5 m lane-center separation so their camera-relative geometry is symmetric.
