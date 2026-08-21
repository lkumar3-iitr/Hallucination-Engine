"""
carla_ego_initialization.py

Shared deterministic CARLA ego initialization.

Purpose
-------
Remove avoidable initial-state differences before a paired
CARLA-vs-HE closed-loop experiment.

Protocol
--------
1. Let the spawned vehicle physically settle onto the road.
2. Read the physically settled z / pitch / roll.
3. Restore exact nominal spawn x / y / yaw.
4. Zero linear and angular velocity.
5. Hold vehicle stationary for several synchronous ticks.
6. Snap once more to the canonical state.
7. Release the hand brake WITHOUT advancing simulation.

The experiment should begin immediately after this function.

This does NOT constrain the vehicle after frame 0.
Once the experiment begins, CARLA and HE are free to diverge
naturally through closed-loop model behavior.
"""

from __future__ import annotations

import math


def vehicle_speed_mps(actor):

    velocity = actor.get_velocity()

    return math.sqrt(
        velocity.x * velocity.x
        + velocity.y * velocity.y
        + velocity.z * velocity.z
    )


def get_vehicle_state(actor):

    transform = actor.get_transform()

    velocity = actor.get_velocity()

    angular_velocity = (
        actor.get_angular_velocity()
    )

    return {

        "x":
            float(
                transform.location.x
            ),

        "y":
            float(
                transform.location.y
            ),

        "z":
            float(
                transform.location.z
            ),

        "pitch":
            float(
                transform.rotation.pitch
            ),

        "yaw":
            float(
                transform.rotation.yaw
            ),

        "roll":
            float(
                transform.rotation.roll
            ),

        "vx":
            float(
                velocity.x
            ),

        "vy":
            float(
                velocity.y
            ),

        "vz":
            float(
                velocity.z
            ),

        "angular_vx":
            float(
                angular_velocity.x
            ),

        "angular_vy":
            float(
                angular_velocity.y
            ),

        "angular_vz":
            float(
                angular_velocity.z
            ),

        "speed_mps":
            float(
                vehicle_speed_mps(
                    actor
                )
            ),
    }


def print_vehicle_state(
    label,
    state,
):

    print(
        f"{label} "
        f"x={state['x']:.6f} "
        f"y={state['y']:.6f} "
        f"z={state['z']:.6f} "
        f"pitch={state['pitch']:.4f} "
        f"yaw={state['yaw']:.4f} "
        f"roll={state['roll']:.4f} "
        f"| "
        f"v=("
        f"{state['vx']:.6f},"
        f"{state['vy']:.6f},"
        f"{state['vz']:.6f}"
        f") "
        f"speed={state['speed_mps']:.6f} "
        f"| "
        f"omega=("
        f"{state['angular_vx']:.6f},"
        f"{state['angular_vy']:.6f},"
        f"{state['angular_vz']:.6f}"
        f")"
    )


def canonicalize_ego_start(
    world,
    ego,
    nominal_spawn_tf,
    settle_ticks=30,
    hold_ticks=5,
):

    import carla

    zero_vector = carla.Vector3D(
        x=0.0,
        y=0.0,
        z=0.0,
    )

    hold_control = carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=1.0,
        hand_brake=True,
        reverse=False,
    )

    neutral_control = carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
    )

    # ========================================================
    # Phase 1:
    # allow gravity / suspension / wheels to physically settle
    # ========================================================

    print(
        "[init] physics settling:",
        settle_ticks,
        "ticks",
    )

    for _ in range(
        settle_ticks
    ):

        ego.apply_control(
            hold_control
        )

        world.tick()

    settled_tf = (
        ego.get_transform()
    )

    settled_state = (
        get_vehicle_state(
            ego
        )
    )

    print_vehicle_state(
        "[init settled]",
        settled_state,
    )

    # ========================================================
    # Phase 2:
    #
    # Exact x/y/yaw come from the nominal map spawn.
    #
    # z/pitch/roll come from the physically settled vehicle so
    # the chassis remains aligned with the actual road surface.
    # ========================================================

    canonical_tf = carla.Transform(

        carla.Location(
            x=float(
                nominal_spawn_tf.location.x
            ),
            y=float(
                nominal_spawn_tf.location.y
            ),
            z=float(
                settled_tf.location.z
            ),
        ),

        carla.Rotation(
            pitch=float(
                settled_tf.rotation.pitch
            ),
            yaw=float(
                nominal_spawn_tf.rotation.yaw
            ),
            roll=float(
                settled_tf.rotation.roll
            ),
        ),
    )

    # ========================================================
    # Phase 3:
    # repeatedly hold exact canonical state
    # ========================================================

    print(
        "[init] canonical hold:",
        hold_ticks,
        "ticks",
    )

    for _ in range(
        hold_ticks
    ):

        ego.set_transform(
            canonical_tf
        )

        ego.set_target_velocity(
            zero_vector
        )

        ego.set_target_angular_velocity(
            zero_vector
        )

        ego.apply_control(
            hold_control
        )

        world.tick()

    # ========================================================
    # Phase 4:
    # final exact snap AFTER the last physics tick
    # ========================================================

    ego.set_transform(
        canonical_tf
    )

    ego.set_target_velocity(
        zero_vector
    )

    ego.set_target_angular_velocity(
        zero_vector
    )

    ego.apply_control(
        hold_control
    )

    canonical_state = (
        get_vehicle_state(
            ego
        )
    )

    print_vehicle_state(
        "[init canonical]",
        canonical_state,
    )

    # ========================================================
    # Release brake without ticking.
    #
    # Therefore no uncontrolled motion occurs between the
    # canonical-state measurement and experiment frame 0.
    # ========================================================

    ego.apply_control(
        neutral_control
    )

    return (
        canonical_tf,
        canonical_state,
    )