import cv2
import numpy as np

from HallucinationEngine import HEConfig
from HallucinationEngine.camera_only import CameraOnlyHE


def make_dummy_road_frame(width=1280, height=720):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (70, 70, 70)

    # Sky-ish region
    frame[:320, :] = (120, 140, 160)

    # Road-ish region
    frame[320:, :] = (55, 55, 55)

    # Lane lines
    cv2.line(frame, (500, height), (610, 330), (210, 210, 210), 4)
    cv2.line(frame, (780, height), (670, 330), (210, 210, 210), 4)

    # OpenCV frame is BGR-like, but values are grayish.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    config = HEConfig(
        enabled=True,
        scenario_type="camera_only_wrong_way",

        trigger_frame=10,
        duration_frames=120,

        image_width=1280,
        image_height=720,

        sprite_path="assets/sprites/car_front.png",
        render_mode="sprite",

        road_center_x_ratio=0.5,
        horizon_y_ratio=0.45,
        spawn_y_ratio=0.48,
        target_y_ratio=0.82,

        min_box_width=35,
        max_box_width=360,
        min_box_height=30,
        max_box_height=300,

        sprite_alpha=1.0,
        sprite_horizontal_scale=1.10,
        sprite_vertical_scale=1.15,
        sprite_y_offset_ratio=0.0,

        match_brightness=True,
        add_shadow=True,
        soften_edges=True,
        motion_blur=False,
        draw_sprite_debug_box=True,

        use_visual_road_estimator=True,
        road_estimator_roi_y_start_ratio=0.55,
        road_estimator_roi_y_end_ratio=0.95,
        road_center_smoothing_alpha=0.85,
    )

    he = CameraOnlyHE(config)

    dt = 1.0 / 20.0

    for frame_id in range(160):
        frame_rgb = make_dummy_road_frame()

        modified, metadata = he.process(
            frame_rgb=frame_rgb,
            frame_id=frame_id,
            dt=dt,
        )

        he_info = metadata.get("hallucination", {})

        road_estimate = metadata.get("road_estimate", {})

        if he_info.get("active", False):
            print(
                "CameraOnlyHE:",
                "frame =", frame_id,
                "center =", he_info.get("image_center"),
                "box =", he_info.get("box_2d"),
                "progress =", round(he_info.get("progress", 0.0), 3),
                "road_center =", round(road_estimate.get("road_center_x", -1), 2),
                "road_conf =", round(road_estimate.get("confidence", 0.0), 2),
                "road_method =", road_estimate.get("method"),
            )

        display = cv2.cvtColor(modified, cv2.COLOR_RGB2BGR)

        cv2.imshow("CameraOnlyHE Test", display)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()