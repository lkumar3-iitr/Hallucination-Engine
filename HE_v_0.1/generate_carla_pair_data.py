import os
import json
import queue
import random
import traceback
from datetime import datetime

import cv2
import numpy as np
import carla

from HallucinationEngine.insertion import ObjectInserter
from HallucinationEngine.placement.projection_utils import build_camera_intrinsics
from HallucinationEngine import HEConfig
from HallucinationEngine.camera_only import CameraOnlyHE


# ============================================================
# Settings
# ============================================================

HOST = "127.0.0.1"
PORT = 2000
TIMEOUT = 10.0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FOV = 100.0

FIXED_DELTA_SECONDS = 0.05

OUTPUT_BASE_DIR = "paired_data"

SPRITE_PATH = "assets/sprites/carla_front_vehicle.png"

EGO_AUTOPILOT = False

# Default values used when RANDOMIZE_SCENARIO = False
EGO_SPAWN_INDEX = 0
ADVERSARY_FORWARD_M = 40.0
ADVERSARY_LATERAL_M = 0.0

NUM_SEQUENCE_FRAMES = 400
SEQUENCE_FPS = 20

EGO_THROTTLE = 0.25
EGO_STEER = 0.0
EGO_BRAKE = 0.0

USE_MANUAL_EGO_CONTROL = True


# ============================================================
# Randomization settings
# ============================================================

RANDOMIZE_SCENARIO = True

NUM_RANDOM_RUNS = 5

EGO_SPAWN_INDEX_LIST = [
    0, 5, 10, 15, 20, 25, 30, 35
]

ADVERSARY_FORWARD_RANGE_M = (25.0, 30.0)

# Keep small for Stage 1 stationary wrong-way case.
ADVERSARY_LATERAL_RANGE_M = (-0.1, 0.1)

EGO_THROTTLE_RANGE = (0.08, 0.22)

RANDOMIZE_WEATHER = True


# ============================================================
# Basic utilities
# ============================================================

def carla_image_to_rgb_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()


def save_rgb(path, frame_rgb):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frame_bgr)


def flush_image_queue(image_queue):
    while not image_queue.empty():
        try:
            image_queue.get_nowait()
        except queue.Empty:
            break


def setup_synchronous_mode(world):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)


def restore_world_settings(world, original_settings):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def location_to_dict(location):
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def rotation_to_dict(rotation):
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def transform_to_dict(transform):
    return {
        "location": location_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation),
    }


def vector_to_dict(vector):
    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z),
    }


def get_speed_mps(actor):
    velocity = actor.get_velocity()
    return float(np.sqrt(
        velocity.x * velocity.x +
        velocity.y * velocity.y +
        velocity.z * velocity.z
    ))


def distance_2d(loc_a, loc_b):
    dx = float(loc_a.x - loc_b.x)
    dy = float(loc_a.y - loc_b.y)
    return float(np.sqrt(dx * dx + dy * dy))


def forward_displacement_from_start(start_transform, current_location):
    """
    Compute ego forward displacement along its initial heading.

    This gives an approximate ego motion scalar useful for the
    distance-aware box predictor.
    """

    start_loc = start_transform.location
    start_yaw_rad = np.deg2rad(start_transform.rotation.yaw)

    forward_x = np.cos(start_yaw_rad)
    forward_y = np.sin(start_yaw_rad)

    dx = current_location.x - start_loc.x
    dy = current_location.y - start_loc.y

    displacement = dx * forward_x + dy * forward_y

    return float(displacement)


# ============================================================
# Scenario randomization
# ============================================================

def sample_scenario_config(run_id=0):
    """
    Sample one randomized paired-data scenario.

    Stage 1:
        stationary wrong-way front-facing adversary.
    """

    if not RANDOMIZE_SCENARIO:
        return {
            "run_id": run_id,
            "ego_spawn_index": EGO_SPAWN_INDEX,
            "adversary_forward_m": ADVERSARY_FORWARD_M,
            "adversary_lateral_m": ADVERSARY_LATERAL_M,
            "ego_throttle": EGO_THROTTLE,
        }

    scenario = {
        "run_id": run_id,
        "ego_spawn_index": random.choice(EGO_SPAWN_INDEX_LIST),
        "adversary_forward_m": random.uniform(
            ADVERSARY_FORWARD_RANGE_M[0],
            ADVERSARY_FORWARD_RANGE_M[1],
        ),
        "adversary_lateral_m": random.uniform(
            ADVERSARY_LATERAL_RANGE_M[0],
            ADVERSARY_LATERAL_RANGE_M[1],
        ),
        "ego_throttle": random.uniform(
            EGO_THROTTLE_RANGE[0],
            EGO_THROTTLE_RANGE[1],
        ),
    }

    return scenario


def sample_weather():
    weather_presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
    ]

    return random.choice(weather_presets)


# ============================================================
# Ego control
# ============================================================

def apply_ego_control(ego_vehicle, ego_throttle=None):
    if not USE_MANUAL_EGO_CONTROL:
        return

    if ego_throttle is None:
        ego_throttle = EGO_THROTTLE

    control = carla.VehicleControl(
        throttle=float(ego_throttle),
        steer=EGO_STEER,
        brake=EGO_BRAKE,
    )

    ego_vehicle.apply_control(control)


# ============================================================
# Actor spawning
# ============================================================

def spawn_ego_vehicle(world, blueprint_library, spawn_index=0):
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available.")

    spawn_index = int(spawn_index) % len(spawn_points)
    spawn_transform = spawn_points[spawn_index]

    ego = world.try_spawn_actor(vehicle_bp, spawn_transform)

    if ego is None:
        raise RuntimeError(f"Could not spawn ego at spawn index {spawn_index}")

    print("[PairGen] Spawned ego at index:", spawn_index)
    print("[PairGen] Ego spawn:", spawn_transform.location)

    return ego, spawn_transform


def spawn_rgb_camera(world, blueprint_library, ego_vehicle):
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", str(CAMERA_FOV))

    camera_transform = carla.Transform(
        carla.Location(x=0.35, y=-0.30, z=1.40),
        carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=ego_vehicle,
    )

    return camera


def spawn_adversary_vehicle(
    world,
    blueprint_library,
    ego_vehicle,
    adversary_forward_m=None,
    adversary_lateral_m=None,
):
    """
    Spawn a stationary wrong-way adversary in front of ego camera.

    Stage-1 target:
        - place adversary along ego forward/camera direction
        - keep it near the middle of ego view
        - face it toward ego
        - let physics settle on road
        - then freeze it
    """

    if adversary_forward_m is None:
        adversary_forward_m = ADVERSARY_FORWARD_M

    if adversary_lateral_m is None:
        adversary_lateral_m = ADVERSARY_LATERAL_M

    world_map = world.get_map()

    ego_transform = ego_vehicle.get_transform()
    ego_loc = ego_transform.location
    ego_yaw = ego_transform.rotation.yaw
    ego_yaw_rad = np.deg2rad(ego_yaw)

    forward_x = np.cos(ego_yaw_rad)
    forward_y = np.sin(ego_yaw_rad)

    right_x = np.sin(ego_yaw_rad)
    right_y = -np.cos(ego_yaw_rad)

    desired_location = carla.Location(
        x=ego_loc.x + adversary_forward_m * forward_x + adversary_lateral_m * right_x,
        y=ego_loc.y + adversary_forward_m * forward_y + adversary_lateral_m * right_y,
        z=ego_loc.z + 0.5,
    )

    nearest_wp = world_map.get_waypoint(
        desired_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if nearest_wp is None:
        raise RuntimeError("Could not find road waypoint near desired adversary location.")

    spawn_location = carla.Location(
        x=desired_location.x,
        y=desired_location.y,
        z=nearest_wp.transform.location.z + 0.25,
    )

    spawn_rotation = carla.Rotation(
        pitch=0.0,
        yaw=ego_yaw + 180.0,
        roll=0.0,
    )

    adv_transform = carla.Transform(spawn_location, spawn_rotation)

    adv_bp = blueprint_library.find("vehicle.tesla.model3")

    if adv_bp.has_attribute("role_name"):
        adv_bp.set_attribute("role_name", "he_real_adversary")

    print("[PairGen] Ego location:", ego_loc)
    print("[PairGen] Ego yaw:", ego_yaw)
    print("[PairGen] Desired adversary location:", desired_location)
    print("[PairGen] Nearest road waypoint:", nearest_wp.transform.location)
    print("[PairGen] Final adversary spawn transform:", adv_transform)

    adv_vehicle = world.try_spawn_actor(adv_bp, adv_transform)

    if adv_vehicle is None:
        raise RuntimeError(
            "Could not spawn adversary at ego-forward road-center location. "
            "Try reducing adversary_forward_m or changing ego spawn index."
        )

    adv_vehicle.set_autopilot(False)
    adv_vehicle.set_simulate_physics(True)

    for _ in range(30):
        adv_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
            )
        )
        world.tick()

    adv_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    adv_vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    adv_vehicle.set_simulate_physics(False)

    final_transform = adv_vehicle.get_transform()

    print("[PairGen] Spawned real adversary at:", final_transform.location)
    print("[PairGen] Final adversary yaw:", final_transform.rotation.yaw)

    return adv_vehicle, final_transform, adv_bp.id


# ============================================================
# Projection utilities
# ============================================================

def project_carla_world_to_image_local(world_location, camera_actor, K):
    camera_transform = camera_actor.get_transform()
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    point_world = np.array([
        world_location.x,
        world_location.y,
        world_location.z,
        1.0,
    ])

    point_camera = world_2_camera @ point_world

    x_cv = point_camera[1]
    y_cv = -point_camera[2]
    z_cv = point_camera[0]

    if z_cv <= 0.1:
        return None

    point_img = K @ np.array([x_cv, y_cv, z_cv])

    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]

    return float(u), float(v), float(z_cv)


def compute_actor_2d_bbox(actor, camera_actor, camera_intrinsics, image_width, image_height):
    bbox = actor.bounding_box
    actor_transform = actor.get_transform()

    vertices = bbox.get_world_vertices(actor_transform)

    projected_points = []

    for vertex in vertices:
        projected = project_carla_world_to_image_local(
            world_location=vertex,
            camera_actor=camera_actor,
            K=camera_intrinsics,
        )

        if projected is None:
            continue

        u, v, depth = projected

        if u < -image_width or u > 2 * image_width:
            continue

        if v < -image_height or v > 2 * image_height:
            continue

        projected_points.append((u, v))

    if len(projected_points) < 2:
        return None

    xs = [p[0] for p in projected_points]
    ys = [p[1] for p in projected_points]

    x1 = max(0, min(image_width - 1, min(xs)))
    y1 = max(0, min(image_height - 1, min(ys)))
    x2 = max(0, min(image_width - 1, max(xs)))
    y2 = max(0, min(image_height - 1, max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [int(x1), int(y1), int(x2), int(y2)]


# ============================================================
# Scene setup / cleanup
# ============================================================

def setup_world_and_ego(client, spawn_index, weather=None):
    print("[PairGen] Reloading world...")

    world = client.reload_world(False)
    original_settings = world.get_settings()

    setup_synchronous_mode(world)

    if weather is not None:
        world.set_weather(weather)

    blueprint_library = world.get_blueprint_library()

    ego_vehicle, ego_spawn_transform = spawn_ego_vehicle(
        world=world,
        blueprint_library=blueprint_library,
        spawn_index=spawn_index,
    )

    ego_vehicle.set_autopilot(False)

    rgb_camera = spawn_rgb_camera(
        world=world,
        blueprint_library=blueprint_library,
        ego_vehicle=ego_vehicle,
    )

    image_queue = queue.Queue()
    rgb_camera.listen(lambda image: image_queue.put(image))

    camera_intrinsics = build_camera_intrinsics(
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_degrees=CAMERA_FOV,
    )

    for _ in range(10):
        world.tick()
        flush_image_queue(image_queue)

    return {
        "world": world,
        "original_settings": original_settings,
        "blueprint_library": blueprint_library,
        "ego_vehicle": ego_vehicle,
        "ego_spawn_transform": ego_spawn_transform,
        "rgb_camera": rgb_camera,
        "image_queue": image_queue,
        "camera_intrinsics": camera_intrinsics,
        "actors": [ego_vehicle, rgb_camera],
    }


def cleanup_scene(scene):
    if scene is None:
        return

    actors = scene.get("actors", [])

    for actor in actors:
        try:
            if actor is not None and actor.type_id.startswith("sensor."):
                actor.stop()
        except Exception:
            pass

    for actor in actors:
        try:
            if actor is not None:
                actor.destroy()
        except Exception:
            pass

    try:
        restore_world_settings(
            scene.get("world", None),
            scene.get("original_settings", None),
        )
    except Exception:
        pass


# ============================================================
# Sequence capture
# ============================================================

def capture_sequence(
    world,
    image_queue,
    ego_vehicle,
    rgb_camera,
    output_dir,
    num_frames,
    mode_name,
    adv_vehicle=None,
    camera_intrinsics=None,
    ego_throttle=None,
    ego_start_transform=None,
    initial_adversary_distance_m=None,
):
    """
    Capture a sequence.

    New v2 metadata:
        numeric ego pose
        numeric velocity
        ego_speed_mps
        ego_forward_displacement_m
        adversary pose
        distance_to_adversary_m
        distance_progress
    """

    sequence_metadata = []

    flush_image_queue(image_queue)

    if ego_start_transform is None:
        ego_start_transform = ego_vehicle.get_transform()

    for frame_id in range(num_frames):
        apply_ego_control(ego_vehicle, ego_throttle=ego_throttle)

        world.tick()

        try:
            image = image_queue.get(timeout=3.0)
        except queue.Empty:
            print(f"[PairGen] Warning: missing image at frame {frame_id}")
            continue

        frame_rgb = carla_image_to_rgb_array(image)

        frame_path = os.path.join(
            output_dir,
            f"frame_{frame_id:06d}.png",
        )

        save_rgb(frame_path, frame_rgb)

        bbox_2d = None

        if adv_vehicle is not None and camera_intrinsics is not None:
            bbox_2d = compute_actor_2d_bbox(
                actor=adv_vehicle,
                camera_actor=rgb_camera,
                camera_intrinsics=camera_intrinsics,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
            )

        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed_mps = get_speed_mps(ego_vehicle)

        ego_forward_disp_m = forward_displacement_from_start(
            start_transform=ego_start_transform,
            current_location=ego_location,
        )

        adversary_transform_dict = None
        adversary_location_dict = None
        adversary_velocity_dict = None
        distance_to_adversary_m = None
        distance_progress = None

        if adv_vehicle is not None:
            adv_transform = adv_vehicle.get_transform()
            adv_location = adv_transform.location
            adv_velocity = adv_vehicle.get_velocity()

            adversary_transform_dict = transform_to_dict(adv_transform)
            adversary_location_dict = location_to_dict(adv_location)
            adversary_velocity_dict = vector_to_dict(adv_velocity)

            distance_to_adversary_m = distance_2d(ego_location, adv_location)

            if initial_adversary_distance_m is not None and initial_adversary_distance_m > 1e-6:
                distance_progress = 1.0 - (
                    distance_to_adversary_m / float(initial_adversary_distance_m)
                )
                distance_progress = float(np.clip(distance_progress, 0.0, 1.5))

        if adv_vehicle is not None and frame_id % 5 == 0:
            print(
                f"[PairGen] real bbox frame {frame_id}:",
                bbox_2d,
                "| dist:",
                distance_to_adversary_m,
                "| speed:",
                ego_speed_mps,
            )

        meta = {
            "frame": frame_id,
            "mode": mode_name,
            "frame_path": frame_path,

            "ego_transform": str(ego_transform),
            "ego_velocity": str(ego_velocity),

            "ego_transform_numeric": transform_to_dict(ego_transform),
            "ego_location": location_to_dict(ego_location),
            "ego_velocity_numeric": vector_to_dict(ego_velocity),
            "ego_speed_mps": ego_speed_mps,
            "ego_forward_displacement_m": ego_forward_disp_m,

            "adversary_box_2d": bbox_2d,
            "adversary_transform_numeric": adversary_transform_dict,
            "adversary_location": adversary_location_dict,
            "adversary_velocity_numeric": adversary_velocity_dict,

            "initial_adversary_distance_m": initial_adversary_distance_m,
            "distance_to_adversary_m": distance_to_adversary_m,
            "distance_progress": distance_progress,
        }

        sequence_metadata.append(meta)

        if frame_id % 10 == 0:
            print(f"[PairGen] {mode_name}: captured frame {frame_id}/{num_frames}")

    return sequence_metadata


# ============================================================
# HE sequence generation
# ============================================================

def generate_he_oracle_box_sequence(
    clean_dir,
    he_oracle_dir,
    real_sequence_metadata,
):
    inserter = ObjectInserter()
    he_metadata = []

    for item in real_sequence_metadata:
        frame_id = item["frame"]
        box_2d = item.get("adversary_box_2d", None)

        clean_path = os.path.join(
            clean_dir,
            f"frame_{frame_id:06d}.png",
        )

        clean_bgr = cv2.imread(clean_path)

        if clean_bgr is None:
            print("[PairGen] Missing clean frame:", clean_path)
            continue

        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        if box_2d is None:
            he_rgb = clean_rgb
            insertion_status = "no_real_box"
        else:
            he_rgb = inserter.insert_sprite(
                frame_rgb=clean_rgb,
                sprite_path=SPRITE_PATH,
                box_2d=box_2d,
                alpha=1.0,
                horizontal_scale=1.1,
                vertical_scale=1.15,
                y_offset_ratio=0.0,
                match_brightness=True,
                add_shadow=True,
                soften_edges=True,
                motion_blur=False,
                debug_box=False,
            )
            insertion_status = "inserted_using_real_box"

        he_path = os.path.join(
            he_oracle_dir,
            f"frame_{frame_id:06d}.png",
        )

        save_rgb(he_path, he_rgb)

        he_metadata.append({
            "frame": frame_id,
            "he_frame_path": he_path,
            "box_2d": box_2d,
            "insertion_status": insertion_status,
            "sprite_path": SPRITE_PATH,
            "method": "oracle_box_appearance_only",
        })

        if frame_id % 10 == 0:
            print(f"[PairGen] he_oracle_box: generated frame {frame_id}/{len(real_sequence_metadata)}")

    return he_metadata


def generate_he_camera_only_sequence(clean_dir, he_camera_only_dir, num_frames):
    config = HEConfig(
        enabled=True,
        scenario_type="camera_only_wrong_way",

        trigger_frame=0,
        duration_frames=num_frames - 1,

        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,

        sprite_path=SPRITE_PATH,
        render_mode="sprite",

        road_center_x_ratio=0.5,
        horizon_y_ratio=0.45,
        spawn_y_ratio=0.55,
        target_y_ratio=0.88,

        use_visual_road_estimator=True,
        road_estimator_roi_y_start_ratio=0.55,
        road_estimator_roi_y_end_ratio=0.95,
        road_center_smoothing_alpha=0.75,

        min_box_width=70,
        max_box_width=520,
        min_box_height=55,
        max_box_height=430,

        repeat_after_finish=False,
        repeat_gap_frames=30,

        road_follow_base_gain=0.12,
        road_follow_conf_gain=0.35,

        sprite_alpha=1.0,
        sprite_horizontal_scale=1.10,
        sprite_vertical_scale=1.15,
        sprite_y_offset_ratio=0.0,

        match_brightness=True,
        add_shadow=True,
        soften_edges=True,
        motion_blur=False,
        draw_sprite_debug_box=False,
    )

    he = CameraOnlyHE(config)

    he_metadata = []
    dt = FIXED_DELTA_SECONDS

    for frame_id in range(num_frames):
        clean_path = os.path.join(
            clean_dir,
            f"frame_{frame_id:06d}.png",
        )

        clean_bgr = cv2.imread(clean_path)

        if clean_bgr is None:
            print("[PairGen] Missing clean frame:", clean_path)
            continue

        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        he_rgb, metadata = he.process(
            frame_rgb=clean_rgb,
            frame_id=frame_id,
            dt=dt,
        )

        he_path = os.path.join(
            he_camera_only_dir,
            f"frame_{frame_id:06d}.png",
        )

        save_rgb(he_path, he_rgb)

        he_info = metadata.get("hallucination", {})

        he_metadata.append({
            "frame": frame_id,
            "he_frame_path": he_path,
            "box_2d": he_info.get("box_2d", None),
            "image_center": he_info.get("image_center", None),
            "progress": he_info.get("progress", None),
            "active": he_info.get("active", False),
            "road_estimate": metadata.get("road_estimate", None),
            "method": "camera_only_he_no_real_box",
        })

        if frame_id % 10 == 0:
            print(f"[PairGen] he_camera_only: generated frame {frame_id}/{num_frames}")

    return he_metadata


# ============================================================
# Output directories
# ============================================================

def create_run_dirs(run_id=0):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.abspath(
        os.path.join(OUTPUT_BASE_DIR, f"pair_run_{timestamp}_r{run_id:02d}")
    )

    dirs = {
        "run": run_dir,
        "clean": os.path.join(run_dir, "clean"),
        "real_object": os.path.join(run_dir, "real_object"),
        "he_oracle_box": os.path.join(run_dir, "he_oracle_box"),
        "he_camera_only": os.path.join(run_dir, "he_camera_only"),
        "metadata": os.path.join(run_dir, "metadata"),
    }

    print("[PairGen] Creating output folders...")

    for name, d in dirs.items():
        os.makedirs(d, exist_ok=True)
        print(f"[PairGen] Created {name}: {d}")

    return dirs


# ============================================================
# One paired run
# ============================================================

def run_one_pair_generation(client, run_id=0):
    clean_scene = None
    real_scene = None

    scenario_config = sample_scenario_config(run_id=run_id)
    scenario_weather = sample_weather() if RANDOMIZE_WEATHER else None

    print("[PairGen] Sampled scenario config:")
    print(json.dumps(scenario_config, indent=4))
    print("[PairGen] Weather:", scenario_weather)

    dirs = create_run_dirs(run_id=run_id)
    print("[PairGen] Output:", dirs["run"])

    adv_bp_id = None
    adv_transform = None
    initial_distance_to_adversary_m = None

    try:
        # --------------------------------------------------------
        # 1. Clean sequence in fresh world
        # --------------------------------------------------------
        print("[PairGen] Setting up clean scene...")

        clean_scene = setup_world_and_ego(
            client=client,
            spawn_index=scenario_config["ego_spawn_index"],
            weather=scenario_weather,
        )

        clean_ego_start_transform = clean_scene["ego_vehicle"].get_transform()

        print("[PairGen] Capturing clean sequence...")

        clean_sequence_metadata = capture_sequence(
            world=clean_scene["world"],
            image_queue=clean_scene["image_queue"],
            ego_vehicle=clean_scene["ego_vehicle"],
            rgb_camera=clean_scene["rgb_camera"],
            output_dir=dirs["clean"],
            num_frames=NUM_SEQUENCE_FRAMES,
            mode_name="clean",
            adv_vehicle=None,
            camera_intrinsics=None,
            ego_throttle=scenario_config["ego_throttle"],
            ego_start_transform=clean_ego_start_transform,
            initial_adversary_distance_m=scenario_config["adversary_forward_m"],
        )

        cleanup_scene(clean_scene)
        clean_scene = None

        # --------------------------------------------------------
        # 2. Real-object sequence in fresh world
        # --------------------------------------------------------
        print("[PairGen] Setting up real-object scene...")

        real_scene = setup_world_and_ego(
            client=client,
            spawn_index=scenario_config["ego_spawn_index"],
            weather=scenario_weather,
        )

        real_ego_start_transform = real_scene["ego_vehicle"].get_transform()

        print("[PairGen] Spawning real adversary...")

        adv_vehicle, adv_transform, adv_bp_id = spawn_adversary_vehicle(
            world=real_scene["world"],
            blueprint_library=real_scene["blueprint_library"],
            ego_vehicle=real_scene["ego_vehicle"],
            adversary_forward_m=scenario_config["adversary_forward_m"],
            adversary_lateral_m=scenario_config["adversary_lateral_m"],
        )

        real_scene["actors"].append(adv_vehicle)

        initial_distance_to_adversary_m = distance_2d(
            real_scene["ego_vehicle"].get_location(),
            adv_vehicle.get_location(),
        )

        print("[PairGen] Initial distance to adversary:", initial_distance_to_adversary_m)

        for _ in range(30):
            real_scene["world"].tick()
            flush_image_queue(real_scene["image_queue"])

        print("[PairGen] Capturing real-object sequence...")

        real_sequence_metadata = capture_sequence(
            world=real_scene["world"],
            image_queue=real_scene["image_queue"],
            ego_vehicle=real_scene["ego_vehicle"],
            rgb_camera=real_scene["rgb_camera"],
            output_dir=dirs["real_object"],
            num_frames=NUM_SEQUENCE_FRAMES,
            mode_name="real_object",
            adv_vehicle=adv_vehicle,
            camera_intrinsics=real_scene["camera_intrinsics"],
            ego_throttle=scenario_config["ego_throttle"],
            ego_start_transform=real_ego_start_transform,
            initial_adversary_distance_m=initial_distance_to_adversary_m,
        )

        cleanup_scene(real_scene)
        real_scene = None

        # --------------------------------------------------------
        # 3. Generate HE oracle-box sequence
        # --------------------------------------------------------
        print("[PairGen] Generating HE oracle-box sequence...")

        he_oracle_metadata = generate_he_oracle_box_sequence(
            clean_dir=dirs["clean"],
            he_oracle_dir=dirs["he_oracle_box"],
            real_sequence_metadata=real_sequence_metadata,
        )

        # --------------------------------------------------------
        # 4. Generate HE camera-only sequence
        # --------------------------------------------------------
        print("[PairGen] Generating HE camera-only sequence...")

        he_camera_only_metadata = generate_he_camera_only_sequence(
            clean_dir=dirs["clean"],
            he_camera_only_dir=dirs["he_camera_only"],
            num_frames=NUM_SEQUENCE_FRAMES,
        )

        # --------------------------------------------------------
        # 5. Save metadata
        # --------------------------------------------------------
        sequence_metadata = {
            "num_frames": NUM_SEQUENCE_FRAMES,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "sequence_fps": SEQUENCE_FPS,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "camera_fov": CAMERA_FOV,

            "scenario_config": scenario_config,
            "weather": str(scenario_weather),

            "purpose": {
                "clean": "input stream without adversary",
                "real_object": "reference sequence with actual CARLA adversary",
                "he_oracle_box": "diagnostic appearance-only HE using real projected box",
                "he_camera_only": "old procedural camera-only HE baseline",
            },

            "clean_sequence": clean_sequence_metadata,
            "real_sequence": real_sequence_metadata,
            "he_oracle_box_sequence": he_oracle_metadata,
            "he_camera_only_sequence": he_camera_only_metadata,

            "adversary": {
                "blueprint_id": adv_bp_id,
                "forward_m": scenario_config["adversary_forward_m"],
                "lateral_m": scenario_config["adversary_lateral_m"],
                "initial_distance_to_adversary_m": initial_distance_to_adversary_m,
                "transform": str(adv_transform),
                "transform_numeric": transform_to_dict(adv_transform) if adv_transform is not None else None,
            },

            "ego": {
                "spawn_index": scenario_config["ego_spawn_index"],
                "ego_throttle": scenario_config["ego_throttle"],
            },
        }

        metadata_path = os.path.join(
            dirs["metadata"],
            "sequence_metadata.json",
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(sequence_metadata, f, indent=4)

        print("[PairGen] Sequence metadata saved:", metadata_path)

    except Exception:
        traceback.print_exc()

    finally:
        print("[PairGen] Cleaning up run...")

        cleanup_scene(clean_scene)
        cleanup_scene(real_scene)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[PairGen] Run done.")


# ============================================================
# Main
# ============================================================

def main():
    print("[PairGen] Starting paired-data generation script...")

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)

    for run_id in range(NUM_RANDOM_RUNS):
        print("=" * 80)
        print(f"[PairGen] Starting run {run_id + 1}/{NUM_RANDOM_RUNS}")
        print("=" * 80)

        run_one_pair_generation(
            client=client,
            run_id=run_id,
        )

    print("[PairGen] All requested runs finished.")


if __name__ == "__main__":
    main()