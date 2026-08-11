import math


class LaneAwarePlacement:
    """
    Chooses a hallucinated object location using route waypoints.

    This version is CARLA-compatible but does not import carla.
    It expects CARLA-like waypoint/transform objects.
    """

    def __init__(self, config):
        self.config = config

    def choose_location_ahead(self, metadata, distance_ahead_m):
        """
        Select a point ahead of the ego vehicle along the route.

        Required metadata:
            ego_transform
            route_waypoints

        Returns:
            dict with world_xyz, yaw, selected waypoint
        """

        ego_transform = metadata.get("ego_transform", None)
        route_waypoints = metadata.get("route_waypoints", None)

        if ego_transform is None or route_waypoints is None:
            return None

        if len(route_waypoints) == 0:
            return None

        ego_location = ego_transform.location

        closest_index = self._find_closest_waypoint_index(
            ego_location,
            route_waypoints
        )

        if closest_index is None:
            return None

        target_wp = self._find_waypoint_at_distance(
            route_waypoints,
            closest_index,
            distance_ahead_m
        )

        if target_wp is None:
            return None

        wp_transform = target_wp.transform
        loc = wp_transform.location
        yaw = wp_transform.rotation.yaw

        # For wrong-way vehicle, heading should be opposite to route direction.
        adversary_yaw = self._normalize_angle(yaw + 180.0)

        # Optional lateral offset.
        # Positive means shift to waypoint's right side.
        shifted_x, shifted_y = self._apply_lateral_offset(
            loc.x,
            loc.y,
            yaw,
            self.config.lateral_offset_m
        )

        return {
            "world_xyz": [shifted_x, shifted_y, loc.z],
            "yaw": adversary_yaw,
            "waypoint": target_wp,
            "lane_id": getattr(target_wp, "lane_id", None),
            "road_id": getattr(target_wp, "road_id", None),
        }

    def _find_closest_waypoint_index(self, ego_location, route_waypoints):
        best_index = None
        best_distance = float("inf")

        for i, wp in enumerate(route_waypoints):
            loc = wp.transform.location

            dx = loc.x - ego_location.x
            dy = loc.y - ego_location.y
            dz = loc.z - ego_location.z

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < best_distance:
                best_distance = dist
                best_index = i

        return best_index

    def _find_waypoint_at_distance(self, route_waypoints, start_index, distance_ahead_m):
        accumulated = 0.0

        prev_loc = route_waypoints[start_index].transform.location

        for i in range(start_index + 1, len(route_waypoints)):
            loc = route_waypoints[i].transform.location

            dx = loc.x - prev_loc.x
            dy = loc.y - prev_loc.y
            dz = loc.z - prev_loc.z

            step_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            accumulated += step_dist

            if accumulated >= distance_ahead_m:
                return route_waypoints[i]

            prev_loc = loc

        return None

    def _apply_lateral_offset(self, x, y, yaw_degrees, offset_m):
        """
        Shift point sideways relative to waypoint heading.

        Forward direction:
            [cos(yaw), sin(yaw)]

        Right direction:
            [sin(yaw), -cos(yaw)]
        """

        if abs(offset_m) < 1e-6:
            return x, y

        yaw = math.radians(yaw_degrees)

        right_x = math.sin(yaw)
        right_y = -math.cos(yaw)

        shifted_x = x + offset_m * right_x
        shifted_y = y + offset_m * right_y

        return shifted_x, shifted_y

    def _normalize_angle(self, angle):
        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle