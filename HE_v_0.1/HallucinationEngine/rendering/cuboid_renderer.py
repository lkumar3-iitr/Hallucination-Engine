import cv2

from HallucinationEngine.placement.projection_utils import (
    build_3d_vehicle_corners,
    project_world_to_image,
)


class CuboidRenderer:
    """
    Debug renderer that draws projected 3D cuboid edges.
    """

    def __init__(self, config):
        self.config = config

    def render(self, frame, hallucination, metadata=None):
        if hallucination is None:
            return frame

        if not hallucination.get("active", False):
            return frame

        if metadata is None:
            return frame

        world_xyz = hallucination.get("world_xyz", None)
        yaw = hallucination.get("yaw", None)

        camera_transform = metadata.get("camera_transform", None)
        camera_intrinsics = metadata.get("camera_intrinsics", None)

        if world_xyz is None or yaw is None:
            return frame

        if camera_transform is None or camera_intrinsics is None:
            return frame

        vehicle_center_xyz = [
            world_xyz[0],
            world_xyz[1],
            world_xyz[2] + self.config.vehicle_height_m * 0.5
        ]

        corners_3d = build_3d_vehicle_corners(
            center_xyz=vehicle_center_xyz,
            yaw_degrees=yaw,
            length_m=self.config.vehicle_length_m,
            width_m=self.config.vehicle_width_m,
            height_m=self.config.vehicle_height_m,
        )

        projected = []

        for corner in corners_3d:
            point = project_world_to_image(
                world_location=corner,
                camera_transform=camera_transform,
                K=camera_intrinsics,
            )

            if point is None:
                projected.append(None)
                continue

            u, v, depth = point

            if depth <= 0.1:
                projected.append(None)
                continue

            projected.append((int(u), int(v)))

        modified = frame.copy()

        color = self.config.box_color
        thickness = self.config.box_thickness

        # Corner index layout:
        # 0-3 top rectangle, 4-7 bottom rectangle
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for i, j in edges:
            if projected[i] is None or projected[j] is None:
                continue

            cv2.line(
                modified,
                projected[i],
                projected[j],
                color,
                thickness,
                cv2.LINE_AA
            )

        box = hallucination.get("box_2d", None)

        if box is not None:
            x1, y1, x2, y2 = box

            cv2.rectangle(
                modified,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

        label = hallucination.get("scenario_type", "HE cuboid")

        if box is not None:
            x1, y1, _, _ = box
            cv2.putText(
                modified,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return modified