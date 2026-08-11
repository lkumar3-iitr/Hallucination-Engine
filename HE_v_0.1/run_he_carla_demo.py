import os
import sys
import time
import queue
import random
import traceback

import cv2
import numpy as np

import carla

from HallucinationEngine import HallucinationEngine, HEConfig
from HallucinationEngine.placement.projection_utils import build_camera_intrinsics
from HallucinationEngine.metadata import HERecorder
from HallucinationEngine.camera_only import CameraOnlyHE
import json
# ============================================================
# Basic settings
# ============================================================

HOST = "127.0.0.1"
PORT = 2000
TIMEOUT = 10.0

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FOV = 100.0

FIXED_DELTA_SECONDS = 0.05  # 20 FPS

DISPLAY_WINDOW_NAME = "CARLA + Hallucination Engine"
CONFIG_PATH = "configs/he_demo_config.json"
# ============================================================
# Saving settings
# ============================================================

SAVE_HE_FRAMES = False       # Set True only when you want PNG frames
SAVE_HE_METADATA = False      # JSON metadata is small
SAVE_ONLY_ACTIVE = False      # Save only frames where HE is active
# ============================================================
# Utility functions
# ============================================================
def load_demo_config(config_path):
    """
    Load HE demo configuration from JSON file.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config
def carla_image_to_rgb_array(image):
    """
    Convert CARLA sensor.camera.rgb image to RGB numpy array.

    CARLA raw image format is BGRA.
    Output is RGB.
    """

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # BGRA -> RGB
    rgb = array[:, :, :3][:, :, ::-1]

    return rgb.copy()


def generate_route_waypoints_ahead(vehicle, world_map, step_distance=2.0, max_distance=120.0):
    """
    Generate a simple route ahead of the ego vehicle using CARLA map waypoints.

    This avoids needing a GlobalRoutePlanner for the first HE test.

    Args:
        vehicle: ego vehicle actor
        world_map: CARLA map
        step_distance: distance between route waypoints
        max_distance: how far ahead to generate route

    Returns:
        list of CARLA waypoints
    """

    vehicle_location = vehicle.get_location()

    current_wp = world_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if current_wp is None:
        return []

    route = [current_wp]

    distance_accumulated = 0.0

    while distance_accumulated < max_distance:
        next_wps = current_wp.next(step_distance)

        if not next_wps:
            break

        # For this first demo, just choose the first available continuation.
        current_wp = next_wps[0]
        route.append(current_wp)

        distance_accumulated += step_distance

    return route


def setup_synchronous_mode(world, traffic_manager=None, fixed_delta_seconds=0.05):
    """
    Enable synchronous simulation.
    """

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    if traffic_manager is not None:
        traffic_manager.set_synchronous_mode(True)


def restore_world_settings(world, original_settings, traffic_manager=None):
    """
    Restore CARLA world settings.
    """

    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)

    if traffic_manager is not None:
        traffic_manager.set_synchronous_mode(False)


def spawn_ego_vehicle(world, blueprint_library):
    """
    Spawn a simple ego vehicle.
    """

    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available in this CARLA map.")

    random.shuffle(spawn_points)

    ego_vehicle = None

    for spawn_point in spawn_points:
        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if ego_vehicle is not None:
            print("Spawned ego vehicle at:", spawn_point.location)
            break

    if ego_vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle.")

    return ego_vehicle


def spawn_rgb_camera(world, blueprint_library, ego_vehicle, camera_config):
    """
    Spawn RGB camera attached to ego vehicle.
    """

    camera_bp = blueprint_library.find("sensor.camera.rgb")

    image_width = int(camera_config["image_width"])
    image_height = int(camera_config["image_height"])
    camera_fov = float(camera_config["fov"])

    camera_bp.set_attribute("image_size_x", str(image_width))
    camera_bp.set_attribute("image_size_y", str(image_height))
    camera_bp.set_attribute("fov", str(camera_fov))

    camera_transform = carla.Transform(
        carla.Location(
            x=float(camera_config["x"]),
            y=float(camera_config["y"]),
            z=float(camera_config["z"])
        ),
        carla.Rotation(
            pitch=float(camera_config["pitch"]),
            yaw=float(camera_config["yaw"]),
            roll=float(camera_config["roll"])
        )
    )

    rgb_camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=ego_vehicle
    )

    return rgb_camera
def get_he_mode(he_config_dict):
    """
    Return HE mode.

    Supported:
        oracle_geometry
        camera_only
    """

    return he_config_dict.get("he_mode", "oracle_geometry")
def init_hallucination_engine(he_config_dict, camera_config):
    """
    Initialize HE using external config dictionary.
    """

    he_config = HEConfig(
        enabled=bool(he_config_dict["enabled"]),

        scenario_type=he_config_dict["scenario_type"],

        trigger_frame=int(he_config_dict["trigger_frame"]),
        duration_frames=int(he_config_dict["duration_frames"]),

        initial_distance_m=float(he_config_dict["initial_distance_m"]),
        min_distance_m=float(he_config_dict["min_distance_m"]),
        relative_speed_mps=float(he_config_dict["relative_speed_mps"]),

        use_ego_speed_for_relative_motion=bool(
            he_config_dict.get("use_ego_speed_for_relative_motion", True)
        ),
        adversary_speed_mps=float(
            he_config_dict.get(
                "adversary_speed_mps",
                he_config_dict.get("relative_speed_mps", 5.0)
            )
        ),
        max_closing_speed_mps=float(
            he_config_dict.get("max_closing_speed_mps", 25.0)
        ),

        lateral_offset_m=float(he_config_dict["lateral_offset_m"]),

        vehicle_length_m=float(he_config_dict["vehicle_length_m"]),
        vehicle_width_m=float(he_config_dict["vehicle_width_m"]),
        vehicle_height_m=float(he_config_dict["vehicle_height_m"]),

        render_mode=he_config_dict["render_mode"],

        # Main/default sprite settings
        sprite_path=he_config_dict.get(
            "sprite_path",
            "assets/sprites/car_front.png"
        ),
        sprite_alpha=float(
            he_config_dict.get("sprite_alpha", 1.0)
        ),
        sprite_vertical_scale=float(
            he_config_dict.get("sprite_vertical_scale", 1.15)
        ),
        sprite_horizontal_scale=float(
            he_config_dict.get("sprite_horizontal_scale", 1.10)
        ),
        sprite_y_offset_ratio=float(
            he_config_dict.get("sprite_y_offset_ratio", 0.0)
        ),
        draw_sprite_debug_box=bool(
            he_config_dict.get("draw_sprite_debug_box", False)
        ),

        # Scenario-specific sprites.
        # For now both can point to the same car_front.png.
        wrong_way_sprite_path=he_config_dict.get(
            "wrong_way_sprite_path",
            he_config_dict.get("sprite_path", "assets/sprites/car_front.png")
        ),
        stopped_vehicle_sprite_path=he_config_dict.get(
            "stopped_vehicle_sprite_path",
            he_config_dict.get("sprite_path", "assets/sprites/car_front.png")
        ),

        # ObjectInserter realism options
        match_brightness=bool(
            he_config_dict.get("match_brightness", True)
        ),
        add_shadow=bool(
            he_config_dict.get("add_shadow", True)
        ),
        soften_edges=bool(
            he_config_dict.get("soften_edges", True)
        ),
        motion_blur=bool(
            he_config_dict.get("motion_blur", False)
        ),

        

        # Camera-only settings
        road_center_x_ratio=float(
            he_config_dict.get("road_center_x_ratio", 0.5)
        ),
        horizon_y_ratio=float(
            he_config_dict.get("horizon_y_ratio", 0.45)
        ),
        spawn_y_ratio=float(
            he_config_dict.get("spawn_y_ratio", 0.48)
        ),
        target_y_ratio=float(
            he_config_dict.get("target_y_ratio", 0.82)
        ),

        min_box_width=int(
            he_config_dict.get("min_box_width", 35)
        ),
        max_box_width=int(
            he_config_dict.get("max_box_width", 360)
        ),
        min_box_height=int(
            he_config_dict.get("min_box_height", 30)
        ),
        max_box_height=int(
            he_config_dict.get("max_box_height", 300)
        ),

        use_visual_road_estimator=bool(
            he_config_dict.get("use_visual_road_estimator", True)
        ),
        road_estimator_roi_y_start_ratio=float(
            he_config_dict.get("road_estimator_roi_y_start_ratio", 0.55)
        ),
        road_estimator_roi_y_end_ratio=float(
            he_config_dict.get("road_estimator_roi_y_end_ratio", 0.95)
        ),
        road_center_smoothing_alpha=float(
            he_config_dict.get("road_center_smoothing_alpha", 0.85)
        ),

        image_width=int(camera_config["image_width"]),
        image_height=int(camera_config["image_height"]),
    )

    he_mode = get_he_mode(he_config_dict)

    if he_mode == "camera_only":
        return CameraOnlyHE(he_config)

    if he_mode == "oracle_geometry":
        return HallucinationEngine(he_config)

    raise ValueError(f"Unknown he_mode: {he_mode}")

def draw_debug_text(frame_rgb, text_lines):
    """
    Draw debug text on RGB frame.
    """

    frame = frame_rgb.copy()

    y = 30

    for line in text_lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 28

    return frame


# ============================================================
# Main demo
# ============================================================

def main():
    client = None
    world = None
    original_settings = None
    traffic_manager = None

    actors_to_destroy = []

    image_queue = queue.Queue()

    try:
        demo_config = load_demo_config(CONFIG_PATH)
        print("Loaded config from:", CONFIG_PATH)
        print("Top-level config keys:", demo_config.keys())
        sim_config = demo_config["simulation"]
        camera_config = demo_config["camera"]
        he_config_dict = demo_config["hallucination"]
        saving_config = demo_config.get("saving", {
            "base_dir": "he_outputs",
            "save_only_active": True,
            "save_frames": False,
            "save_metadata": True
        })
        client = carla.Client(HOST, PORT)
        client.set_timeout(float(sim_config["timeout"]))

        world = client.get_world()
        original_settings = world.get_settings()

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        setup_synchronous_mode(
            world,
            traffic_manager,
            fixed_delta_seconds=float(sim_config["fixed_delta_seconds"])
        )

        blueprint_library = world.get_blueprint_library()
        world_map = world.get_map()

        ego_vehicle = spawn_ego_vehicle(world, blueprint_library)
        actors_to_destroy.append(ego_vehicle)

        rgb_camera = spawn_rgb_camera(
            world=world,
            blueprint_library=blueprint_library,
            ego_vehicle=ego_vehicle,
            camera_config=camera_config
        )
        actors_to_destroy.append(rgb_camera)

        rgb_camera.listen(lambda image: image_queue.put(image))

        camera_intrinsics = build_camera_intrinsics(
            image_width=int(camera_config["image_width"]),
            image_height=int(camera_config["image_height"]),
            fov_degrees=float(camera_config["fov"])
        )

        print("Camera intrinsics:")
        print(camera_intrinsics)

        hallucination_engine = init_hallucination_engine(
            he_config_dict=he_config_dict,
            camera_config=camera_config
        )
        recorder = HERecorder(
            base_dir=saving_config["base_dir"],
            save_only_active=bool(saving_config["save_only_active"]),
            save_frames=bool(saving_config["save_frames"]),
            save_metadata=bool(saving_config["save_metadata"]),
        )

        print("[HE Save Settings]")
        print("  Save frames:", SAVE_HE_FRAMES)
        print("  Save metadata:", SAVE_HE_METADATA)
        print("  Save only active:", SAVE_ONLY_ACTIVE)
        # Let the vehicle move using CARLA autopilot.
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        frame_id = 0

        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("\nRunning CARLA + HE demo.")
        print("Press Q in the OpenCV window to stop.\n")

        while True:
            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: camera image queue timeout.")
                continue

            rgb_frame = carla_image_to_rgb_array(image)

            he_mode = get_he_mode(he_config_dict)

            if he_mode == "oracle_geometry":
                route_waypoints = generate_route_waypoints_ahead(
                    vehicle=ego_vehicle,
                    world_map=world_map,
                    step_distance=2.0,
                    max_distance=120.0
                )
            else:
                route_waypoints = []

            metadata = {
                "frame": frame_id,
                "dt": float(sim_config["fixed_delta_seconds"]),

                "ego_transform": ego_vehicle.get_transform(),
                "ego_velocity": ego_vehicle.get_velocity(),

                "camera_transform": rgb_camera.get_transform(),
                "camera_intrinsics": camera_intrinsics,

                "route_waypoints": route_waypoints,
            }

            he_mode = get_he_mode(he_config_dict)

            if he_mode == "camera_only":
                modified_frame, updated_metadata = hallucination_engine.process(
                    frame_rgb=rgb_frame,
                    frame_id=frame_id,
                    dt=float(sim_config["fixed_delta_seconds"])
                )
            else:
                modified_frame, updated_metadata = hallucination_engine.process(
                    frame=rgb_frame,
                    metadata=metadata
                )
            recorder.record(
                frame_rgb=modified_frame,
                updated_metadata=updated_metadata
            )
            he_info = updated_metadata.get("hallucination", {})
            
            status = he_info.get("projection_status", "inactive")
            distance = he_info.get("distance_to_ego_m", None)
            box = he_info.get("box_2d", None)

            debug_lines = [
                f"Frame: {frame_id}",
                f"Route waypoints: {len(route_waypoints)}",
                f"HE active: {he_info.get('active', False)}",
                f"HE status: {status}",
            ]
            road_estimate = updated_metadata.get("road_estimate", None)

            if road_estimate is not None:
                debug_lines.append(
                    f"Road center: {road_estimate.get('road_center_x', -1):.1f}"
                )
                debug_lines.append(
                    f"Road conf: {road_estimate.get('confidence', 0.0):.2f}"
                )
                debug_lines.append(
                    f"Road method: {road_estimate.get('method', 'NA')}"
                )
            if distance is not None:
                debug_lines.append(f"HE distance: {distance:.2f} m")

            if box is not None:
                debug_lines.append(f"HE box: {box}")

            if he_info.get("approach_speed_mps", None) is not None:
                debug_lines.append(f"HE approach speed: {he_info['approach_speed_mps']:.2f} m/s")
            if he_info.get("ego_forward_speed_mps", None) is not None:
                debug_lines.append(f"Ego forward speed: {he_info['ego_forward_speed_mps']:.2f} m/s")

            if he_info.get("adversary_speed_mps", None) is not None:
                debug_lines.append(f"Adv speed: {he_info['adversary_speed_mps']:.2f} m/s")

            if he_info.get("relative_speed_mps", None) is not None:
                debug_lines.append(f"Closing speed: {he_info['relative_speed_mps']:.2f} m/s")

            if he_info.get("age_frames", None) is not None:
                debug_lines.append(f"HE age: {he_info['age_frames']} frames")
            modified_frame = draw_debug_text(modified_frame, debug_lines)

            if he_info.get("active", False):
                if he_mode == "camera_only":
                    print(
                        "CameraOnlyHE:",
                        "frame =", frame_id,
                        "| center =", he_info.get("image_center"),
                        "| box =", he_info.get("box_2d"),
                        "| progress =", round(he_info.get("progress", 0.0), 3),
                        "| road_center =",
                        round(
                            updated_metadata.get("road_estimate", {}).get("road_center_x", -1),
                            2
                        ),
                        "| road_conf =",
                        round(
                            updated_metadata.get("road_estimate", {}).get("confidence", 0.0),
                            2
                        ),
                    )
                else:
                    print(
                        "HE:",
                        "status =", he_info.get("projection_status"),
                        "| dist =", round(he_info.get("distance_to_ego_m", -1), 2),
                        "| approach_speed =", round(he_info.get("approach_speed_mps", -1), 2),
                        "| ego_speed =", round(he_info.get("ego_forward_speed_mps", -1), 2),
                        "| adv_speed =", round(he_info.get("adversary_speed_mps", -1), 2),
                        "| closing =", round(he_info.get("relative_speed_mps", -1), 2),
                        "| age =", he_info.get("age_frames"),
                        "| box =", he_info.get("box_2d"),
                        "| lane =", he_info.get("lane_id"),
                        "| road =", he_info.get("road_id"),
                    )

            # OpenCV expects BGR for display.
            display_frame = cv2.cvtColor(modified_frame, cv2.COLOR_RGB2BGR)

            cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception:
        traceback.print_exc()

    finally:
        print("Cleaning up...")
        try:
            if "recorder" in locals() and recorder is not None:
                recorder.save_summary()
        except Exception as e:
            print("Could not save HE recorder summary:", e)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        for actor in actors_to_destroy:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        restore_world_settings(world, original_settings, traffic_manager)

        print("Done.")


if __name__ == "__main__":
    main()