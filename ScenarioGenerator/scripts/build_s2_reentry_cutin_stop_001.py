import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT
    / "outputs"
    / "v2_resolved"
    / "partial_occlusion_cutin_001.resolved_v2.json"
)

DST = (
    ROOT
    / "outputs"
    / "v2_resolved"
    / "s2_reentry_cutin_stop_001.resolved_v2.json"
)


FPS = 20
DURATION = 32.0
FRAMES = int(DURATION * FPS) + 1

X0 = 12.0
LEFT_Y = -2.8

V_SLOW = 3.0
V_FAST = 9.0

T_PULL_OUT = 5.0
T_LEFT = 7.5

T_BRAKE1_END = 8.5

T_RESTART = 15.0
T_FAST = 17.25

T_CUTIN = 23.0
T_MERGED = 25.0

T_BRAKE2_END = 27.25


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return 3.0 * u * u - 2.0 * u * u * u


def smoothstep_rate(u, duration):
    u = max(0.0, min(1.0, u))
    return (
        (6.0 * u - 6.0 * u * u)
        / duration
    )


# ------------------------------------------------------------
# Important longitudinal anchor positions
# ------------------------------------------------------------

X_LEFT = (
    X0
    + V_SLOW * T_LEFT
)

# Brake 3 -> 0 in 1 second.
A_BRAKE1 = 3.0

X_STOP1 = (
    X_LEFT
    + V_SLOW * (T_BRAKE1_END - T_LEFT)
    - 0.5
    * A_BRAKE1
    * (T_BRAKE1_END - T_LEFT) ** 2
)

# Accelerate 0 -> 9 m/s at 4 m/s^2.
A_FAST = 4.0

X_FAST = (
    X_STOP1
    + 0.5
    * A_FAST
    * (T_FAST - T_RESTART) ** 2
)

X_CUTIN = (
    X_FAST
    + V_FAST * (T_CUTIN - T_FAST)
)

X_MERGED = (
    X_CUTIN
    + V_FAST * (T_MERGED - T_CUTIN)
)

X_STOP2 = (
    X_MERGED
    + V_FAST * (T_BRAKE2_END - T_MERGED)
    - 0.5
    * A_FAST
    * (T_BRAKE2_END - T_MERGED) ** 2
)


def actor_state(t):

    # --------------------------------------------------------
    # 0–5 s:
    # slow lead, ego lane
    # --------------------------------------------------------

    if t < T_PULL_OUT:

        x = X0 + V_SLOW * t
        y = 0.0

        vx = V_SLOW
        vy = 0.0

    # --------------------------------------------------------
    # 5–7.5 s:
    # smooth move into left lane
    # --------------------------------------------------------

    elif t < T_LEFT:

        dt = t - T_PULL_OUT

        u = (
            dt
            / (T_LEFT - T_PULL_OUT)
        )

        s = smoothstep(u)

        x = X0 + V_SLOW * t
        y = LEFT_Y * s

        vx = V_SLOW

        vy = (
            LEFT_Y
            * smoothstep_rate(
                u,
                T_LEFT - T_PULL_OUT,
            )
        )

    # --------------------------------------------------------
    # 7.5–8.5 s:
    # brake to stop in left lane
    # --------------------------------------------------------

    elif t < T_BRAKE1_END:

        dt = t - T_LEFT

        vx = max(
            0.0,
            V_SLOW - A_BRAKE1 * dt,
        )

        x = (
            X_LEFT
            + V_SLOW * dt
            - 0.5 * A_BRAKE1 * dt * dt
        )

        y = LEFT_Y
        vy = 0.0

    # --------------------------------------------------------
    # 8.5–15 s:
    # stationary in left lane
    # --------------------------------------------------------

    elif t < T_RESTART:

        x = X_STOP1
        y = LEFT_Y

        vx = 0.0
        vy = 0.0

    # --------------------------------------------------------
    # 15–17.25 s:
    # accelerate hard from behind
    # --------------------------------------------------------

    elif t < T_FAST:

        dt = t - T_RESTART

        vx = min(
            V_FAST,
            A_FAST * dt,
        )

        x = (
            X_STOP1
            + 0.5 * A_FAST * dt * dt
        )

        y = LEFT_Y
        vy = 0.0

    # --------------------------------------------------------
    # 17.25–23 s:
    # high-speed catch / overtake
    # --------------------------------------------------------

    elif t < T_CUTIN:

        x = (
            X_FAST
            + V_FAST * (t - T_FAST)
        )

        y = LEFT_Y

        vx = V_FAST
        vy = 0.0

    # --------------------------------------------------------
    # 23–25 s:
    # high-speed cut-in back to ego lane
    # --------------------------------------------------------

    elif t < T_MERGED:

        dt = t - T_CUTIN

        u = (
            dt
            / (T_MERGED - T_CUTIN)
        )

        s = smoothstep(u)

        x = (
            X_CUTIN
            + V_FAST * dt
        )

        y = (
            LEFT_Y
            * (1.0 - s)
        )

        vx = V_FAST

        vy = (
            -LEFT_Y
            * smoothstep_rate(
                u,
                T_MERGED - T_CUTIN,
            )
        )

    # --------------------------------------------------------
    # 25–27.25 s:
    # emergency braking in ego lane
    # --------------------------------------------------------

    elif t < T_BRAKE2_END:

        dt = t - T_MERGED

        vx = max(
            0.0,
            V_FAST - A_FAST * dt,
        )

        x = (
            X_MERGED
            + V_FAST * dt
            - 0.5 * A_FAST * dt * dt
        )

        y = 0.0
        vy = 0.0

    # --------------------------------------------------------
    # 27.25–32 s:
    # stopped in ego lane
    # --------------------------------------------------------

    else:

        x = X_STOP2
        y = 0.0

        vx = 0.0
        vy = 0.0

    speed = math.hypot(
        vx,
        vy,
    )

    if speed > 1e-6:

        yaw = math.degrees(
            math.atan2(
                vy,
                vx,
            )
        )

    else:
        yaw = 0.0

    return (
        x,
        y,
        yaw,
        speed,
        vx,
        vy,
    )


with SRC.open(
    "r",
    encoding="utf-8",
) as f:

    data = json.load(f)


data["scenario_id"] = (
    "s2_reentry_cutin_stop_001"
)

data["source_description"] = (
    "S2 long-form re-entry cut-in safety scenario. "
    "A slow Nissan Patrol begins ahead of the ego in the same lane, "
    "moves into the adjacent left lane and stops, allowing the ego "
    "to pass. The same Nissan later accelerates from behind at high "
    "speed, overtakes in the left lane, cuts back into the ego lane, "
    "then performs hard braking to a complete stop."
)

data["duration_s"] = DURATION
data["fps"] = FPS

data["actors"] = [
    {
        "actor_id":
            "adv_reentry",

        "actor_type":
            "vehicle",

        "role":
            "adversary",

        "asset_key":
            "vehicle.passenger_02",

        "dimensions_m":
            None,

        "spawn_time_s":
            0.0,

        "despawn_time_s":
            None,
    }
]


# Ego trajectory is only a schema placeholder.
# The generic closed-loop runner drives the real ego.

data["ego_frames"] = [
    {
        "frame_idx": i,
        "t_s": i / FPS,

        "x_m": 0.0,
        "y_m": 0.0,
        "yaw_deg": 0.0,

        "speed_mps": 0.0,
        "vx_mps": 0.0,
        "vy_mps": 0.0,
    }
    for i in range(FRAMES)
]


actor_frames = []

for i in range(FRAMES):

    t = i / FPS

    (
        x,
        y,
        yaw,
        speed,
        vx,
        vy,
    ) = actor_state(t)

    actor_frames.append({
        "frame_idx": i,
        "t_s": t,

        "actor_id":
            "adv_reentry",

        "x_m":
            float(x),

        "y_m":
            float(y),

        "yaw_deg":
            float(yaw),

        "speed_mps":
            float(speed),

        "vx_mps":
            float(vx),

        "vy_mps":
            float(vy),
    })


data["actor_frames"] = (
    actor_frames
)


with DST.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        data,
        f,
        indent=2,
    )


print("created:", DST)
print("frames:", FRAMES)

print(
    "first stop:",
    round(X_STOP1, 3),
)

print(
    "cut-in starts:",
    round(X_CUTIN, 3),
)

print(
    "second stop:",
    round(X_STOP2, 3),
)