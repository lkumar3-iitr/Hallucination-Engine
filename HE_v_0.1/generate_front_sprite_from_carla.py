import os
import queue
import random
import traceback

import cv2
import numpy as np
import carla


# ============================================================
# Settings
# ============================================================

HOST = "127.0.0.1"
PORT = 2000
TIMEOUT = 10.0

FIXED_DELTA_SECONDS = 0.05

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_FOV = 55.0

OUTPUT_DIR = "assets/sprites"
OUTPUT_SPRITE_PATH = os.path.join(OUTPUT_DIR, "carla_front_vehicle.png")
OUTPUT_DEBUG_PATH = os.path.join(OUTPUT_DIR, "carla_front_vehicle_debug.png")

VEHICLE_BLUEPRINT_ID = "vehicle.tesla.model3"

# Camera configuration for front-facing sprite generation
CAMERA_DISTANCE_M = 10.0
CAMERA_HEIGHT_M = 1.35

# Crop margin around final extracted object
CROP_MARGIN_PX = 15


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_synchronous_mode(world):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)


def restore_world_settings(world, original_settings):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def safe_destroy(actor):
    if actor is None:
        return

    try:
        if actor.type_id.startswith("sensor."):
            actor.stop()
    except Exception:
        pass

    try:
        actor.destroy()
    except Exception:
        pass


def carla_rgb_to_array(image):
    """
    Convert CARLA RGB image to RGB numpy array.
    CARLA raw buffer is BGRA.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()


def get_latest_sensor_frame(world, q, ticks=1):
    flush_queue(q)

    latest = None

    for _ in range(ticks):
        world.tick()
        try:
            latest = q.get(timeout=3.0)
        except queue.Empty:
            print("[SpriteGen] Warning: no sensor frame received.")

    if latest is None:
        raise RuntimeError("No sensor frame received.")

    return latest


# ============================================================
# Camera geometry
# ============================================================

def build_camera_intrinsics(width, height, fov_degrees):
    """
    Build camera intrinsic matrix.
    """
    fov = np.deg2rad(fov_degrees)
    focal = width / (2.0 * np.tan(fov / 2.0))

    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def project_carla_world_to_image(world_location, camera_actor, K):
    """
    Project CARLA world location to image.

    CARLA camera coordinates:
        x = forward
        y = right
        z = up

    OpenCV camera coordinates:
        x = right
        y = down
        z = forward

    Conversion:
        [x_cv, y_cv, z_cv] = [y_carla, -z_carla, x_carla]
    """

    camera_transform = camera_actor.get_transform()
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    point_world = np.array([
        world_location.x,
        world_location.y,
        world_location.z,
        1.0
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


def compute_actor_2d_bbox(actor, camera_actor, K, image_width, image_height):
    """
    Project CARLA actor 3D bbox into image.
    """
    bbox = actor.bounding_box
    actor_transform = actor.get_transform()

    vertices = bbox.get_world_vertices(actor_transform)

    projected_points = []

    for vertex in vertices:
        p = project_carla_world_to_image(
            world_location=vertex,
            camera_actor=camera_actor,
            K=K
        )

        if p is None:
            continue

        u, v, depth = p

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
# CARLA scene setup
# ============================================================

def spawn_vehicle(world, blueprint_library):
    """
    Spawn vehicle at a valid CARLA spawn point.
    """
    vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT_ID)
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    random.shuffle(spawn_points)

    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.set_simulate_physics(False)

            # Tick once so transform is updated reliably
            world.tick()

            print("[SpriteGen] Spawned vehicle:", VEHICLE_BLUEPRINT_ID)
            print("[SpriteGen] Vehicle location:", vehicle.get_location())
            print("[SpriteGen] Vehicle yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle

    raise RuntimeError("Could not spawn vehicle.")


def spawn_front_camera(world, blueprint_library, vehicle):
    """
    Spawn camera in front of vehicle, looking directly at vehicle center/front.
    """
    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location
    vehicle_yaw = vehicle_transform.rotation.yaw

    yaw_rad = np.deg2rad(vehicle_yaw)

    forward_x = np.cos(yaw_rad)
    forward_y = np.sin(yaw_rad)

    camera_location = carla.Location(
        x=vehicle_loc.x + CAMERA_DISTANCE_M * forward_x,
        y=vehicle_loc.y + CAMERA_DISTANCE_M * forward_y,
        z=vehicle_loc.z + CAMERA_HEIGHT_M,
    )

    # Aim the camera toward vehicle center / upper body
    target_location = carla.Location(
        x=vehicle_loc.x,
        y=vehicle_loc.y,
        z=vehicle_loc.z + 1.0,
    )

    dx = target_location.x - camera_location.x
    dy = target_location.y - camera_location.y
    dz = target_location.z - camera_location.z

    yaw = np.rad2deg(np.arctan2(dy, dx))
    horizontal_dist = np.sqrt(dx * dx + dy * dy)
    pitch = np.rad2deg(np.arctan2(dz, horizontal_dist))

    camera_rotation = carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
    )

    camera_transform = carla.Transform(camera_location, camera_rotation)

    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    rgb_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    rgb_bp.set_attribute("fov", str(CAMERA_FOV))

    rgb_camera = world.spawn_actor(rgb_bp, camera_transform)

    print("[SpriteGen] Spawned front RGB camera at:", camera_location)
    print("[SpriteGen] Camera yaw:", yaw)
    print("[SpriteGen] Camera pitch:", pitch)

    return rgb_camera


# ============================================================
# Mask extraction
# ============================================================

def clean_mask(mask):
    """
    Clean binary mask and keep largest component.
    """
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label

    cleaned = (labels == largest_label).astype(np.uint8) * 255
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)

    return cleaned


def make_mask_from_bbox_grabcut(rgb, bbox_2d):
    """
    Extract vehicle mask using GrabCut initialized from projected 2D bbox.

    Improved version:
        - uses the projected bbox as strong spatial constraint
        - suppresses road/background leakage
        - keeps only largest central component
    """

    image_h, image_w = rgb.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox_2d]

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 5 or box_h <= 5:
        raise RuntimeError("BBox too small for GrabCut.")

    # Smaller expansion than before.
    # Too much expansion allows road/background into the mask.
    margin_x = int(box_w * 0.06)
    margin_y = int(box_h * 0.06)

    x1e = max(0, x1 - margin_x)
    y1e = max(0, y1 - margin_y)
    x2e = min(image_w - 1, x2 + margin_x)
    y2e = min(image_h - 1, y2 + margin_y)

    crop_rgb = rgb[y1e:y2e + 1, x1e:x2e + 1].copy()

    if crop_rgb.size == 0:
        raise RuntimeError("Empty crop for GrabCut.")

    crop_h, crop_w = crop_rgb.shape[:2]

    # Initialize mask as probable background.
    mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Inner rectangle is probable foreground.
    # Keep it smaller than the crop to give GrabCut clear background border.
    fg_margin_x = max(3, int(crop_w * 0.12))
    fg_margin_y = max(3, int(crop_h * 0.10))

    mask[
        fg_margin_y:crop_h - fg_margin_y,
        fg_margin_x:crop_w - fg_margin_x
    ] = cv2.GC_PR_FGD

    # Strong background border.
    border = max(4, int(min(crop_w, crop_h) * 0.04))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    cv2.grabCut(
        crop_bgr,
        mask,
        None,
        bgd_model,
        fgd_model,
        7,
        cv2.GC_INIT_WITH_MASK
    )

    mask_fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Remove very bottom leakage.
    # Vehicle bottom should be near bottom, but not a full road rectangle.
    bottom_cut = int(crop_h * 0.96)
    mask_fg[bottom_cut:, :] = 0

    # Morphological cleanup.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component near crop center.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg)

    if num_labels > 1:
        crop_cx = crop_w / 2.0
        crop_cy = crop_h / 2.0

        best_label = None
        best_score = -1e9

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if area < 300:
                continue

            cx, cy = centroids[label]

            dist = abs(cx - crop_cx) + 0.5 * abs(cy - crop_cy)
            score = area - 4.0 * dist

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None:
            mask_fg = (labels == best_label).astype(np.uint8) * 255

    # Feather alpha edge slightly.
    mask_fg = cv2.GaussianBlur(mask_fg, (3, 3), 0)

    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1e:y2e + 1, x1e:x2e + 1] = mask_fg

    return full_mask


# ============================================================
# Crop and save
# ============================================================

def crop_rgba_from_mask(rgb, mask):
    """
    Create tight RGBA crop around object mask.
    """
    ys, xs = np.where(mask > 10)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Mask is empty. Could not crop sprite.")

    h, w = mask.shape[:2]

    x1 = max(0, int(xs.min()) - CROP_MARGIN_PX)
    y1 = max(0, int(ys.min()) - CROP_MARGIN_PX)
    x2 = min(w - 1, int(xs.max()) + CROP_MARGIN_PX)
    y2 = min(h - 1, int(ys.max()) + CROP_MARGIN_PX)

    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1]
    crop_alpha = mask[y1:y2 + 1, x1:x2 + 1]

    rgba = np.dstack([crop_rgb, crop_alpha])

    return rgba, [x1, y1, x2, y2]


def save_rgba_png(path, rgba):
    """
    Save RGBA image using OpenCV.
    OpenCV expects BGRA.
    """
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, bgra)


def save_debug_image(path, rgb, mask, crop_box, bbox_2d=None):
    """
    Save debug image with mask + crop box + optional projected bbox.
    """
    debug = rgb.copy()

    if bbox_2d is not None:
        bx1, by1, bx2, by2 = bbox_2d
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 3)

    mask_color = np.zeros_like(debug)
    mask_color[:, :, 1] = mask

    debug = cv2.addWeighted(debug, 0.75, mask_color, 0.25, 0)
    debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, debug_bgr)


# ============================================================
# Main
# ============================================================

def main():
    client = None
    world = None
    original_settings = None
    actors = []

    rgb_queue = queue.Queue()

    try:
        ensure_dir(OUTPUT_DIR)

        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT)

        world = client.get_world()
        original_settings = world.get_settings()

        setup_synchronous_mode(world)
        blueprint_library = world.get_blueprint_library()

        vehicle = spawn_vehicle(world, blueprint_library)
        actors.append(vehicle)

        rgb_camera = spawn_front_camera(world, blueprint_library, vehicle)
        actors.append(rgb_camera)

        rgb_camera.listen(lambda image: rgb_queue.put(image))

        # Warm up
        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)

        rgb_image = get_latest_sensor_frame(world=world, q=rgb_queue, ticks=5)
        rgb = carla_rgb_to_array(rgb_image)

        K = build_camera_intrinsics(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            fov_degrees=CAMERA_FOV
        )

        bbox_2d = compute_actor_2d_bbox(
            actor=vehicle,
            camera_actor=rgb_camera,
            K=K,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT
        )

        print("[SpriteGen] Projected vehicle bbox:", bbox_2d)

        if bbox_2d is None:
            raise RuntimeError("Could not project vehicle bbox. Camera may not see vehicle.")

        mask = make_mask_from_bbox_grabcut(
            rgb=rgb,
            bbox_2d=bbox_2d
        )

        print("[SpriteGen] GrabCut mask pixels:", int(np.sum(mask > 0)))

        if np.sum(mask > 0) < 500:
            raise RuntimeError("GrabCut mask too small. Vehicle extraction failed.")

        mask = clean_mask(mask)

        print("[SpriteGen] Cleaned mask pixels:", int(np.sum(mask > 0)))

        rgba, crop_box = crop_rgba_from_mask(rgb, mask)

        save_rgba_png(OUTPUT_SPRITE_PATH, rgba)
        save_debug_image(
            OUTPUT_DEBUG_PATH,
            rgb=rgb,
            mask=mask,
            crop_box=crop_box,
            bbox_2d=bbox_2d
        )

        print("[SpriteGen] Saved sprite:", OUTPUT_SPRITE_PATH)
        print("[SpriteGen] Saved debug:", OUTPUT_DEBUG_PATH)
        print("[SpriteGen] Crop box:", crop_box)
        print("[SpriteGen] Sprite shape:", rgba.shape)

    except Exception:
        traceback.print_exc()

    finally:
        print("[SpriteGen] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)

        restore_world_settings(world, original_settings)

        print("[SpriteGen] Done.")


if __name__ == "__main__":
    main()