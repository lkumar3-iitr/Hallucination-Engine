metadata = {
    "frame": frame_id,
    "dt": FIXED_DELTA_SECONDS,

    "ego_transform": ego_vehicle.get_transform(),
    "ego_velocity": ego_vehicle.get_velocity(),

    "camera_transform": rgb_camera.get_transform(),
    "camera_intrinsics": camera_intrinsics,

    "route_waypoints": route_waypoints,
}