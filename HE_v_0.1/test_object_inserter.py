import cv2
import numpy as np

from HallucinationEngine.insertion import ObjectInserter


def main():
    frame_path = "he_test_frame.png"
    sprite_path = "assets/sprites/car_front.png"

    # If you do not have a test frame yet, create simple road-like frame
    frame = cv2.imread(frame_path)

    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)

        # Draw simple road region
        cv2.line(frame, (480, 720), (620, 320), (120, 120, 120), 4)
        cv2.line(frame, (800, 720), (660, 320), (120, 120, 120), 4)

    # cv2 reads BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inserter = ObjectInserter()

    box_2d = [540, 330, 740, 560]

    output_rgb = inserter.insert_sprite(
        frame_rgb=frame_rgb,
        sprite_path=sprite_path,
        box_2d=box_2d,
        alpha=1.0,
        horizontal_scale=1.1,
        vertical_scale=1.15,
        y_offset_ratio=0.0,
        match_brightness=True,
        add_shadow=True,
        soften_edges=True,
        motion_blur=False,
        debug_box=True,
    )

    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("ObjectInserter Test", output_bgr)
    cv2.imwrite("object_inserter_test_output.png", output_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()